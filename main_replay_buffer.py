import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import logging
from pyspark.sql.functions import lit
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
from tensorflow.keras.optimizers import Adam
import random
from itertools import combinations
from pyspark.sql.functions import to_timestamp
from sklearn.preprocessing import LabelEncoder
from pyspark.ml.feature import IndexToString
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer
from pyspark.context import SparkContext
from pyspark.sql.types import ArrayType, IntegerType, FloatType, StructType, StructField
import time

class ContinualNextItemPredictor:
    def __init__(self, 
                 embedding_dim: int = 32,
                 window_size_rows: int = 30,
                 buffer_size: int = 50):
        
        self.embedding_dim = embedding_dim
        self.window_size_rows = window_size_rows
        
        self.buffer_size = buffer_size
        self.buffer_data = pd.DataFrame()
        
        
        # self.item_encoder = LabelEncoder()
        # self.user_encoder = LabelEncoder()
        # self.session_encoder = LabelEncoder()
        self.item_indexer = StringIndexer(inputCol="product_id", outputCol="product_index", handleInvalid='keep')
        self.user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid='keep')
        self.session_indexer = StringIndexer(inputCol="user_session", outputCol="session_index", handleInvalid='keep')
        
        self.model = None
        self.performance_history = []

        self.n_items = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    
    def initial_train_model(self, df: pd.DataFrame):
        """Update the model with new data."""
        
        # Train the model on the window data
        print("[DEBUG][initial_train_model] starting training")
        self.n_items = self.train(df)
        
        # Log update metrics
        self.logger.info(f"Training data size: {df.count()}")
        self.logger.info(f"Unique products: {df.select('product_id').distinct().count()}")
        self.logger.info(f"Unique sessions: {df.select('user_session').distinct().count()}")
        return self.n_items 
    
    def update_model(self, buffer_data, df: pd.DataFrame):
        """Update the model with new data."""

        # Retrain the model on the window data
        print("[DEBUG][update_model] starting training")
        # train_df = pd.concat([df, buffer_data], ignore_index=True)
        train_df = df.union(buffer_data)
        self.n_items  = self.train(train_df)
        
        # Log update metrics
        self.logger.info(f"Training data size: {train_df.count()}")
        self.logger.info(f"Unique products: {df.select('product_id').distinct().count()}")
        self.logger.info(f"Unique sessions: {df.select('user_session').distinct().count()}")

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
    
    def _transform_with_unknown(self, series: pd.Series, encoder: LabelEncoder) -> np.ndarray:
        """Transform values, mapping unknown values to a special token."""
        # Create a mask for known values
        known_mask = np.isin(series, encoder.classes_)
        
        # Initialize result array with a special token (-1) for unknown values
        result = np.full(len(series), -1)
        
        # Transform known values
        result[known_mask] = encoder.transform(series[known_mask])
        
        return result
    
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
    
    def build_model(self, sequence_length: int = 100):
        """Build the neural network model."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.n_items, self.embedding_dim), # input_length=sequence_length),
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.n_items, activation='softmax')
        ])

        learning_rate = 0.0045
    
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )


        
    def train(self, spark_df, epochs: int = 50, batch_size: int = 128):
        """Train the model using PySpark DataFrame."""
        processed_df = self.preprocess_data(spark_df, fit_encoders=True)
        print('Preprocessing Complete.')
        sequences, targets = self.create_sequences(processed_df)
        print('Sequence creation Complete.')
        if len(sequences) == 0:
            raise ValueError("No valid sequences found for training")
        self.n_items = len(self.item_indexer_model.labels)
        print('N Items = ', self.n_items)
        self.build_model(self.n_items)
        self.model.fit(
            sequences, 
            targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        return self.n_items
    
    def predict_next_items(self, session_history: List[int], k: int = 50, exclude_items: List[int] = None) -> List[Tuple[int, float]]:
        if exclude_items is None:
            exclude_items = set()
        else:
            exclude_items = set(exclude_items)

        # print("[DEBUG] N Items inside predict = ",self.n_items)
        session_history = [item if 0 <= item < self.n_items else 0 for item in session_history]
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


    def evaluate_top_k(self, test_sequences: List[List[int]], test_targets: List[int], k_values: List[int]) -> Dict[int, float]:
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
        # for k in k_values:
        k = 100
        correct_predictions = 0
        total_predictions = 0
        counter = 0
        # For each sequence and its corresponding target
        for seq, target in zip(test_sequences, test_targets):
            counter += 1
            top_k_predictions = self.predict_next_items(seq, k=k)
            predicted_item_ids = [item[0] for item in top_k_predictions]
            if int(target) in predicted_item_ids:
                correct_predictions += 1
            total_predictions += 1

        accuracy = correct_predictions / total_predictions
        accuracies[k] = accuracy

        return accuracies

    
    def evaluate_and_log_performance(self, test_sequences: np.ndarray, 
                                   test_targets: np.ndarray,
                                   current_time: datetime):
        """Evaluate model performance and log metrics."""
        
        print("[DEBUG] starting evaluate top k accuracy")
        top_k_accuracy = self.evaluate_top_k(test_sequences, test_targets, current_time, k=100)
        
        self.performance_history.append({
            'timestamp': current_time,
            'top_k_accuracy': top_k_accuracy
        })
        
        self.logger.info(f"Performance at {current_time}: Top-k accuracy = {top_k_accuracy:.4f}")
        
        # Check for performance degradation
        if len(self.performance_history) >= 2:
            prev_acc = self.performance_history[-2]['top_k_accuracy']
            if top_k_accuracy < prev_acc * 0.9:  # 10% degradation threshold
                self.logger.warning("Significant performance degradation detected!")

    
    def update_buffer(self, buffer_size, buffer_data, train_data):
        df_random_rows = train_data.sample(withReplacement=False, fraction=buffer_size / train_data.count())
        buffer_data = buffer_data.union(df_random_rows)
        return buffer_data
    
    

def main():

    def setup_logger(log_file_path: str):
        logging.basicConfig(filename=log_file_path, level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger


    start = time.time()
    
    # Initialize Spark session
    spark = SparkSession.builder.master("local[*]").appName("buffer").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load data with PySpark
    temp_df = spark.read.csv('dataset_100k.csv', header=True, inferSchema=True)
    N = temp_df.count()
    temp_pd_df = temp_df.withColumn("event_time", to_timestamp("event_time"))

    df = temp_pd_df.toPandas()
    # Initialize predictor
    predictor = ContinualNextItemPredictor()
    logger = setup_logger('output.log')
    initial_train_size = int(0.45 * N)
    print('Initial Train Size: ',initial_train_size)
    train_size = int(0.4 * N)
    buffer_size = int(0.05 * N)
    test_size = int(0.05 * N)
    print('Test Size: ',test_size)
    step = int(0.1 * N)
    print('Step: ', step)

    # Initialize predictor
    predictor = ContinualNextItemPredictor(
        window_size_rows=0.4*N,
    )

    print("[DEBUG] Predictor initialized")

    # Initial training
    print("[DEBUG] Training initial model M1")
    initial_train_data = df[:initial_train_size]
    print('Initial Train Data; ', len(initial_train_data))
    train_spark_df = spark.createDataFrame(initial_train_data)
    predictor.n_items = predictor.initial_train_model(train_spark_df)
    print("[DEBUG] N Items in MAIN: ", predictor.n_items)
    print("[DEBUG] Done")
    print("[DEBUG] Updating initial buffer")
    buffer_data = spark.createDataFrame([], temp_pd_df.schema)
    buffer_data = predictor.update_buffer(buffer_size, buffer_data, temp_pd_df.limit(step + test_size))
    print("[DEBUG] Size of buffer data:", buffer_data.count())
    
    print("[DEBUG] Testing initial model M1")
    initial_test_data = df[initial_train_size:initial_train_size+test_size]
    print('[DEBUG] Intitial Test Data Length: ', len(initial_test_data))
    test_spark_df = spark.createDataFrame(initial_test_data)
    processed_test_data = predictor.preprocess_data(test_spark_df, fit_encoders=False)
    test_sequences, test_targets = predictor.create_sequences(processed_test_data)
    k_values = [1, 5, 10, 20, 50]
    accuracies = predictor.evaluate_top_k(test_sequences, test_targets, k_values)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 30)
    for k, accuracy in accuracies.items():
        print(f"Top-{k} Accuracy: {accuracy:.4%}")
    
    # Print test set statistics
    print("\nTest Set Statistics:")
    print(f"[DEBUG] Total test sequences length: {len(test_sequences)}")
  
    
    
    # Simulate continuous operation
    for current in range(N - initial_train_size - test_size, N, step):
        
        print('[DEBUG] Current Run: ', current)
        # This is will run 5 times
        
        current_train_end = current + test_size
        current_train_start = current_train_end - train_size
        print("[DEBUG] current_train_start, current_train_end: ",current_train_start, current_train_end)
        print("[DEBUG] Starting training")
        df_limited = temp_pd_df.limit(current_train_end)
        df_start = temp_pd_df.limit(current_train_start)
       
        current_train_data = df_limited.subtract(df_start) # df[current_train_start : current_train_end]
        predictor.update_model(buffer_data, current_train_data)
        print("[DEBUG] Done")
        print("[DEBUG] Updating buffer")
        predictor.update_buffer(buffer_size, buffer_data, temp_pd_df.limit(current_train_start))
        print("[DEBUG] Done")

        current_test_start = current_train_end
        current_test_end = current_test_start + test_size
        print("[DEBUG] current_test_start, current_test_end: ",current_test_start, current_test_end)
        df_limited_test = temp_pd_df.limit(current_test_end)
        df_start_test = temp_pd_df.limit(current_test_start)
        current_test_data = df_limited_test.subtract(df_start_test) # df[current_test_start : current_test_end]
        processed_current_test_data = predictor.preprocess_data(current_test_data, fit_encoders=False)
        print("[DEBUG] Preprocessed test data")
        test_sequences, test_targets = predictor.create_sequences(processed_current_test_data)
        print("[DEBUG] Sequences created")
        print("[DEBUG] Starting testing (evaluate top k)")
        k_values = [1, 5, 10, 20, 50]
        accuracies = predictor.evaluate_top_k(test_sequences, test_targets, k_values)
        
        # Print results
        print("\nEvaluation Results:")
        print("-" * 30)
        for k, accuracy in accuracies.items():
            print(f"Top-{k} Accuracy: {accuracy:.4%}")

    end = time.time()
    print("Duration: ", end-start)

    spark.stop()

if __name__ == "__main__":
    main()