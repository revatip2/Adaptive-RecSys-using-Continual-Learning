import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import logging
from pyspark.sql.functions import lit, col, hour, dayofweek
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import random
from pyspark.sql.functions import max
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
                 window_size_days: int = 30,
                 update_frequency_days: int = 7,
                 min_pattern_support: int = 10):
        self.embedding_dim = embedding_dim
        self.window_size_days = window_size_days
        self.update_frequency_days = update_frequency_days
        self.min_pattern_support = min_pattern_support
        
        self.latest_model = None
        
        self.item_indexer = StringIndexer(inputCol="product_id", outputCol="product_index", handleInvalid='keep')
        self.user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid='keep')
        self.session_indexer = StringIndexer(inputCol="user_session", outputCol="session_index", handleInvalid='keep')
        
        self.model = None
        self.pattern_memory = {}  
        self.last_update = None
        self.performance_history = []
        self.new_n_items = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_top_following_items(self, items: List, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get top-k items that frequently follow a given item."""
        if not items:
            return []
        
        counter = defaultdict(int)
        for item in items:
            counter[item] += 1
            
        total = len(items)
        return [(item, count/total) for item, count in 
                sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]
                if count >= self.min_pattern_support]

    def needs_update(self, current_time: datetime) -> bool:
        """Check if model needs updating based on time elapsed."""
        if self.last_update is None:
            return True
        
        time_elapsed = current_time - self.last_update
        return time_elapsed.days >= self.update_frequency_days
    
    def initial_train_model(self, df):
        """Update the model with new data."""

        # current_time = pd.Timestamp(df['event_time'].max())
        current_time = df.agg(max('event_time')).collect()[0][0] 
        print(f"[DEBUG] Current time is {current_time}") 
        # Extract and update patterns
        print(f"[DEBUG][initial_train_model] extracting temporal pattern data for {current_time}")
        self.pattern_memory = self._extract_temporal_patterns(df)
        print('[DEBUG] Temporal Patterns generated.')
        # Retrain the model on the window data
        print("[DEBUG][initial_train_model] starting training")
        print("[DEBUG]Initial Schema: ", df.schema)
        self.train(df)
        
        # Record update time
        self.last_update = current_time
        
        # Log update metrics
        self.logger.info(f"Model updated at {current_time}")
        self.logger.info(f"Training data size: {df.count()}")
        self.logger.info(f"Unique products: {df.select('product_id').distinct().count()}")
        self.logger.info(f"Unique sessions: {df.select('product_id').distinct().count()}")
    
    def update_model(self, df):
        """Update the model with new data."""
        current_time = df.agg(F.max('event_time')).collect()[0][0]

        # Define the window start time (current_time - window_size_days)
        window_start = current_time - timedelta(days=self.window_size_days)

        # Filter the DataFrame for records where event_time >= window_start
        window_data = df.filter(F.col('event_time') >= F.lit(window_start))

        # Optionally, you can create a new DataFrame as a copy (for further use)
        window_data = window_data.alias("window_data")

        print("[DEBUG][update_model] current time:", current_time)
        print("[DEBUG][update_model] window start:", window_start)
        
        # Extract and update patterns
        print("[DEBUG][update_model] extracting temporal pattern data")
        self.pattern_memory = self._extract_temporal_patterns(window_data)
        
        # Retrain the model on the window data
        print("[DEBUG][update_model] starting training")
        self.train(window_data)
        
        # Record update time
        self.last_update = current_time
        
        # Log update metrics
        self.logger.info(f"Model updated at {current_time}")
        self.logger.info(f"Training data size: {window_data.count()}")
        self.logger.info(f"Unique products: {window_data.select('product_id').distinct().count()}")
        self.logger.info(f"Unique sessions: {window_data.select('user_session').distinct().count()}")
    
    def predict_next_items(self, session_history: List[int], k: int = 50, exclude_items: List[int] = None) -> List[Tuple[int, float]]:
        if exclude_items is None:
            exclude_items = set()
        else:
            exclude_items = set(exclude_items)

        # print("[DEBUG] N Items inside predict = ",self.n_items)
        session_history = [item if 0 <= item < self.new_n_items else 0 for item in session_history]
        # Pad session history for prediction
        padded_history = pad_sequences([session_history], maxlen=10, padding='pre')
        predictions = self.model.predict(padded_history, verbose=0)[0]
        # print("[DEBUG] Sorting indices..")
        # Create a list of item indices and their predicted probabilities
        sorted_indices = np.argsort(predictions)[::-1]
        # print("[DEBUG] Finding Top k indices..")
        # Sort by predicted probability in descending order
        top_k_indices = sorted_indices[:k]
        print("[DEBUG] Fetching Probas..")
        # Get the predicted probabilities for the top k items
        items_probs = [(idx, predictions[idx]) for idx in top_k_indices]
        print("[DEBUG] Fetched.")
        return items_probs

        
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
    
    def create_sequences(self, spark_df, sequence_length: int = 10):
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
            session_sequence = [int(item) for item in session_sequence]
            if len(session_sequence) < 2:
                continue  # Skip sessions that are too short
            
            print(f"[DEBUG] Session Sequence: ",session_sequence )
            for i in range(len(session_sequence) - 1):
                # print(i)
                # print("[DEBUG]: i - sequence_length + 1 ", i - sequence_length + 1)
                start_idx = i - sequence_length + 1
                if start_idx < 0:
                    start_idx = 0
                seq = session_sequence[start_idx:i + 1]
                # seq = session_sequence[max(0, i - sequence_length + 1):i + 1]
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
    
    def build_model(self, n_items: int, sequence_length: int = 10):
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

        # Build the model with sample input
        sample_input = tf.zeros((1, sequence_length))  # Create dummy input
        self.model(sample_input)  # This builds the model
        
        self.model.compile(
            optimizer='adam',
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
        
        # Get new vocabulary size
        self.new_n_items = len(self.item_indexer_model.labels)
        print(f"[DEBUG][train] New vocabulary size: {self.new_n_items}")
        
        # If we have a previous model, get its vocabulary size
        if self.latest_model is not None:
            old_n_items = self.latest_model.layers[0].input_dim
            print(f"[DEBUG][train] Previous vocabulary size: {old_n_items}")
        
        # Build the new model
        self.build_model(self.new_n_items)
        
        # Transfer weights carefully
        if self.latest_model is not None:
            print("[DEBUG][train] Transferring weights from previous model")
            for i, (source_layer, target_layer) in enumerate(zip(self.latest_model.layers, self.model.layers)):
                if len(source_layer.weights) > 0:
                    if isinstance(source_layer, tf.keras.layers.Embedding):
                        # For embedding layer, only transfer weights for items that exist in both vocabularies
                        old_weights = source_layer.get_weights()[0]  # Shape: (old_n_items, embedding_dim)
                        new_weights = target_layer.get_weights()[0]  # Shape: (new_n_items, embedding_dim)
                        # Get the minimum vocabulary size
                        min_vocab_size = min(old_n_items, self.new_n_items)                        
                        # Transfer weights for common items
                        new_weights[:min_vocab_size] = old_weights[:min_vocab_size]
                        target_layer.set_weights([new_weights])                        
                        print(f"[DEBUG][train] Transferred embedding weights for {min_vocab_size} items")

                    elif isinstance(source_layer, tf.keras.layers.Dense) and i == len(self.model.layers) - 1:
                        # Handle final dense layer (output layer)
                        old_weights, old_bias = source_layer.get_weights()
                        new_weights = target_layer.get_weights()[0]
                        new_bias = target_layer.get_weights()[1]
                        
                        # Transfer weights for the existing items
                        min_vocab_size = min(old_n_items, self.new_n_items)
                        new_weights[:, :min_vocab_size] = old_weights[:, :min_vocab_size]
                        new_bias[:min_vocab_size] = old_bias[:min_vocab_size]
                        
                        target_layer.set_weights([new_weights, new_bias])
                        print(f"[DEBUG][train] Transferred output layer weights for {min_vocab_size} items")

                    else:
                        # For other layers, transfer weights directly
                        target_layer.set_weights(source_layer.get_weights())
                        print(f"[DEBUG][train] Transferred weights for layer {i}: {source_layer.name}")

        self.model.fit(
            sequences, 
            targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        
        self.latest_model = self.model
        print("[DEBUG][train] Model saved as latest_model")
    

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
        k_values = [1, 5, 10, 20, 50]
        accuracies = self.evaluate_top_k(test_sequences, test_targets, k_values)
        
        self.performance_history.append({
            'timestamp': current_time,
            'top_k_accuracy': accuracies
        })
        
        self.logger.info(f"Performance at {current_time}: Top-k accuracy = {accuracies}")
        
        # Check for performance degradation
        if len(self.performance_history) >= 2:
            prev_acc = self.performance_history[-2]['top_k_accuracy']
            if accuracies < prev_acc * 0.9:  # 10% degradation threshold
                self.logger.warning("Significant performance degradation detected!")


def main():

    start = time.time()
    
    # Initialize Spark session
    spark = SparkSession.builder.master("local[*]").appName("transferlearning").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load data with PySpark
    temp_df = spark.read.csv('7k.csv', header=True, inferSchema=True)
    N = temp_df.count()
    temp_pd_df = temp_df.withColumn("event_time", to_timestamp("event_time"))

    df = temp_pd_df.toPandas()

    # Initialize predictor
    predictor = ContinualNextItemPredictor(
        window_size_days=3,
        update_frequency_days=3,
        min_pattern_support=5
    )
    print("[DEBUG] Predictor initialized")

    # Initial training
    initial_cutoff = (df['event_time'].max() - timedelta(days=14)).normalize()
    print("[DEBUG] initial cutoff:", initial_cutoff)
    train_df = df[df['event_time'] <= initial_cutoff]
    train_spark_df = spark.createDataFrame(train_df)
    predictor.initial_train_model(train_spark_df)
    print("[DEBUG] Trained M1")
    
    # Simulate continuous operation
    test_dates = pd.date_range(
        start=initial_cutoff,
        end=df['event_time'].max(),
        freq='1D'
    )
    test_dates = test_dates[1:]
    print("[DEBUG] test_dates :", test_dates, type(test_dates))

    for current_time in test_dates:
        print("\n\n\n[DEBUG] current_time :", current_time)

        # Get data up to current time
        current_data = temp_pd_df.filter(F.col('event_time') <= current_time)
        
        # Check if update is needed
        if predictor.needs_update(current_time):
            print("[DEBUG] retrain day, retraining")
            # train_df = df[df['event_time'] <= current_time]
            train_df = temp_pd_df.filter(F.col('event_time') <= current_time)
            predictor.update_model(train_df)
            print("[DEBUG] done training")
        
        print("[DEBUG] starting today's predictions")
        print("[DEBUG] preprocessing data")
        processed_test_df = predictor.preprocess_data(current_data, fit_encoders=False)
        print("[DEBUG] creating sequences")
        test_sequences, test_targets = predictor.create_sequences(processed_test_df)
        print("[DEBUG] starting evaluation")
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