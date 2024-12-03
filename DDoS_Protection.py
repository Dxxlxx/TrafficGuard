import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            logging.error("The loaded data is empty. Please check the input file.")
            return None
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error("The input file is empty.")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def create_features(data):
    """Create additional features for anomaly detection."""
    required_columns = ['frame.len', 'ip.src']
    for col in required_columns:
        if col not in data.columns:
            logging.error(f"Missing required column: '{col}'")
            return None

    # Create new features
    data['traffic_volume'] = data['frame.len']
    data['ip_freq'] = data.groupby('ip.src')['ip.src'].transform('count')
    
    # Example of additional features: Protocol frequency
    if 'protocol.type' in data.columns:
        data['protocol_freq'] = data.groupby('protocol.type')['protocol.type'].transform('count')
    
    logging.info("Features created successfully.")
    return data

def scale_data(data, features):
    """Scale the relevant features using MinMaxScaler."""
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[features])
        logging.info("Data scaled successfully.")
        return scaled_data
    except KeyError as e:
        logging.error(f"Error scaling data: Missing feature {e}")
        return None

def detect_anomalies(data, scaled_data, contamination=0.01):
    """Detect anomalies using Isolation Forest."""
    model = IsolationForest(contamination=contamination, random_state=42)
    data['anomaly'] = model.fit_predict(scaled_data)
    logging.info("Anomaly detection completed.")
    return data

def visualize_results(data, save_path="anomaly_detection_results.png"):
    """Visualize the results of the anomaly detection."""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=data, x="traffic_volume", y="ip_freq", hue="anomaly", palette={1: "blue", -1: "red"}, alpha=0.6)
    plt.title("Anomaly Detection in Network Traffic", fontsize=16)
    plt.xlabel("Traffic Volume", fontsize=14)
    plt.ylabel("IP Frequency", fontsize=14)
    plt.axhline(0, color='black', lw=0.8, ls='--')
    plt.axvline(0, color='black', lw=0.8, ls='--')
    plt.legend(title='Anomaly Status', loc='upper right')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.show()

def main(file_path, contamination=0.01, features=None, save_path="anomaly_detection_results.png"):
    """Main function to execute the anomaly detection workflow."""
    data = load_data(file_path)
    
    if data is None:
        logging.error("No data to process. Exiting.")
        return
    
    data = create_features(data)
    if data is None:
        logging.error("Feature creation failed. Exiting.")
        return
    
    if features is None:
        features = ['traffic_volume', 'ip_freq']  # Default features for scaling
    
    scaled_data = scale_data(data, features)
    if scaled_data is None:
        logging.error("Data scaling failed. Exiting.")
        return
    
    data = detect_anomalies(data, scaled_data, contamination)
    
    # Print anomalies detected
    anomalous_data = data[data['anomaly'] == -1]
    logging.info(f"Number of anomalies detected: {len(anomalous_data)}")
    print("Anomalies detected:")
    print(anomalous_data )
    
    # Visualize results
    visualize_results(data, save_path)

# Run the main function
if __name__ == "__main__":
    main("network_traffic.csv", contamination=0.01, features=['traffic_volume', 'ip_freq'], save_path="anomaly_detection_results.png") 
    