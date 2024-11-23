import pandas as pd
import numpy as np
import tensorflow 
from pyspark import SparkContext
import sys
from sklearn.preprocessing import LabelEncoder
import time
from pyspark.ml.feature import StringIndexer
import random
from itertools import combinations
from collections import defaultdict
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from typing import List, Tuple, Dict
from pyspark.ml.feature import IndexToString
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer
from collections import defaultdict
from pyspark.context import SparkContext
from pyspark.sql.types import ArrayType, IntegerType, FloatType, StructType, StructField

class NextItemPredictor:
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.item_indexer = StringIndexer(inputCol="product_id", outputCol="product_index", handleInvalid='keep')
        self.user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid='keep')
        self.session_indexer = StringIndexer(inputCol="user_session", outputCol="session_index", handleInvalid='keep')
        self.item_embeddings = {}
        self.session_histories = defaultdict(list)
        self.model = None
        
    def preprocess_data(self, spark_df, fit_encoders):
        """
        Preprocess the raw e-commerce data using PySpark.
        
        Args:
            spark_df: Input PySpark DataFrame
            fit_encoders: If True, fit the encoders. If False, use existing encoders.
        """
        # Sort by user session and timestamp
        window_spec = Window.partitionBy("user_session").orderBy("event_time")
        spark_df = spark_df.withColumn("event_time", F.col("event_time").cast("timestamp"))
        spark_df = spark_df.withColumn("row_num", F.row_number().over(window_spec))
        
        if fit_encoders:
            # Fit and transform using StringIndexer
            print('Fitting Encoders..')
            item_indexer_model = self.item_indexer.fit(spark_df)
            
            indexed_df = item_indexer_model.transform(spark_df)
            self.user_indexer_model = self.user_indexer.fit(indexed_df)
            indexed_df = self.user_indexer_model.transform(indexed_df)
            self.session_indexer_model = self.session_indexer.fit(indexed_df)
            indexed_df = self.session_indexer_model.transform(indexed_df)

            # Save the fitted model for future use
            self.item_indexer_model = item_indexer_model
            self.item_reverse_indexer = IndexToString(inputCol="product_index", outputCol="product_id", labels=self.item_indexer_model.labels)
        else:
            # Transform using existing encoders (assuming the models are already fitted)
            indexed_df = self.item_indexer_model.transform(spark_df)
            indexed_df = self.user_indexer_model.transform(indexed_df)
            indexed_df = self.session_indexer_model.transform(indexed_df)\
        
        # Drop original columns and keep the indexed ones
        indexed_df = indexed_df.drop("product_id", "user_id", "user_session")
        
        
        return indexed_df

    def train(self, spark_df, epochs: int = 50, batch_size: int = 128):
        """Train the model using PySpark DataFrame."""
        processed_df = self.preprocess_data(spark_df, fit_encoders=True)
        print('Preprocessing Complete.')
        sequences, targets = self.create_sequences(processed_df)
        print('Sequence creation Complete.')
        if len(sequences) == 0:
            raise ValueError("No valid sequences found for training")
        n_items = len(self.item_indexer_model.labels)
        print('N Items = ', n_items)
        self.build_model(n_items)
        self.model.fit(
            sequences, 
            targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        return n_items

    def create_sequences(self, spark_df, sequence_length: int = 100):
        """Create input sequences and target items for training using PySpark."""
        sequences = []
        targets = []
        
        # Create a window specification to handle partitioning by session and ordering by event_time
        window_spec = Window.partitionBy("session_index").orderBy("row_num")
        
        # Create a column to get the next product ID in each session (using lag)
        spark_df = spark_df.withColumn(
            "next_product_id", F.lag("product_index", -1).over(window_spec)
        )
        
        # Filter out sequences with unknown items (-1) or invalid next_product_id
        spark_df = spark_df.filter(
            (F.col("product_index") != -1) & (F.col("next_product_id").isNotNull())
        )
        
        # Collect data by session and create sequences
        session_groups = (
            spark_df.groupBy("session_index")
            .agg(F.collect_list("product_index").alias("product_sequence"))
        )
        
        # Collect the data to the driver as a list of Row objects
        session_data = session_groups.collect()

        for row in session_data:
            session_sequence = row['product_sequence']
            
            if len(session_sequence) < 2:
                continue  # Skip sessions that are too short
                
            for i in range(len(session_sequence) - 1):
                seq = session_sequence[max(0, i - sequence_length + 1):i + 1]
                target = session_sequence[i + 1]
                
                if len(seq) > 0:
                    sequences.append(seq)
                    targets.append(target)
        
        # Pad sequences
        if sequences:  # Check if we have any valid sequences
            sequences = pad_sequences(sequences, maxlen=sequence_length, padding='pre')
            return sequences, np.array(targets)
        else:
            return np.array([]), np.array([])
        
    def build_model(self, n_items: int, sequence_length: int = 100):
        """Build the neural network model."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(n_items, self.embedding_dim, input_length=sequence_length),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(n_items, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def evaluate_top_k(self, n_items, test_sequences: List[List[int]], test_targets: List[int], k_values: List[int]) -> Dict[int, float]:
        """
        Evaluate the model for Top-K accuracy.
        
        Args:
            test_sequences: List of sequences of product indices (from the test set)
            test_targets: List of actual next product indices (targets) corresponding to the sequences
            k_values: List of k values to evaluate for Top-K accuracy

        Returns:
            accuracies: Dictionary where keys are k values and values are the top-k accuracies
        """
        accuracies = {}
        
        # Loop over each k value to compute top-k accuracy
        k = 100
        correct_predictions = 0
        total_predictions = 0
        counter = 0
        # For each sequence and its corresponding target
        for seq, target in zip(test_sequences, test_targets):
            print('Counter: ', counter)
            print('Total Sequences: ', len(test_sequences))
            counter += 1
            top_k_predictions = self.predict_next_items(n_items, seq, k=k)
            predicted_item_ids = [item[0] for item in top_k_predictions]
            if int(target) in predicted_item_ids:
                correct_predictions += 1
            total_predictions += 1
        print('Correct Preds: ', correct_predictions)
        print('Total Preds; ', total_predictions)
        # Calculate accuracy for this k value
        accuracy = correct_predictions / total_predictions
        accuracies[k] = accuracy

        return accuracies

    def predict_next_items(self, n_items,session_history: List[int], k: int = 50, exclude_items: List[int] = None) -> List[Tuple[int, float]]:
        if exclude_items is None:
            exclude_items = set()
        else:
            exclude_items = set(exclude_items)
        session_history = [item if 0 <= item < n_items else 0 for item in session_history]
        # Pad session history for prediction
        padded_history = pad_sequences([session_history], maxlen=10, padding='pre')
        predictions = self.model.predict(padded_history, verbose=0)[0]
        
        # Create a list of item indices and their predicted probabilities
        items_probs = []
        for idx, prob in enumerate(predictions):
            if idx not in exclude_items:
                items_probs.append((idx, prob))  # Store the index and probability
        
        # Sort the items by predicted probability in descending order
        items_probs.sort(key=lambda x: x[1], reverse=True)
        return items_probs[:k]
def main():
    start = time.time()
    # Initialize Spark session
    spark = SparkSession.builder.master("local[*]").appName("new").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    # Load data with PySpark
    temp_df = spark.read.csv('dataset_100k.csv', header=True, inferSchema=True)
    temp_pd_df = temp_df.toPandas()
    # Initialize predictor
    predictor = NextItemPredictor()
    
    # Split data while ensuring all product IDs in test set exist in training set
    n_sessions = len(temp_pd_df['user_session'].unique())
    train_sessions = np.random.choice(
        temp_pd_df['user_session'].unique(), 
        size=int(n_sessions * 0.8), 
        replace=False
    )

    train_pd_df = temp_pd_df[temp_pd_df['user_session'].isin(train_sessions)]
    test_pd_df = temp_pd_df[~temp_pd_df['user_session'].isin(train_sessions)]
  
    train_spark_df = spark.createDataFrame(train_pd_df)
    test_spark_df = spark.createDataFrame(test_pd_df)
  
    n_items = predictor.train(train_spark_df)
    print('Training Complete')

    # testing set
    processed_test_df = predictor.preprocess_data(test_spark_df, fit_encoders=False)
    
    # Create test sequences
    test_sequences, test_targets = predictor.create_sequences(processed_test_df)
   
    k_values = [1, 5, 10, 20, 50]
    accuracies = predictor.evaluate_top_k(n_items, test_sequences, test_targets, k_values)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 30)
    for k, accuracy in accuracies.items():
        print(f"Top-{k} Accuracy: {accuracy:.4%}")
    
    # Print test set statistics
    print("\nTest Set Statistics:")
    print(f"Total test sequences: {len(test_sequences)}")
  
    end = time.time()
    print("Duration: ", end-start)

    spark.stop()

main()