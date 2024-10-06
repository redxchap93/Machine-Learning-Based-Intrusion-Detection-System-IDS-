import os
import subprocess
import sys

# Install required Python libraries if not already installed
def install_libraries():
    print("Installing required libraries...")
    required_libraries = ['scikit-learn', 'pandas', 'numpy', 'scapy', 'joblib']
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + required_libraries)

# Capture network packets using Scapy and save to pcap file
def capture_packets():
    from scapy.all import sniff, wrpcap

    PACKET_COUNT = 1000
    print(f"Capturing {PACKET_COUNT} network packets...")

    packets = sniff(count=PACKET_COUNT)
    wrpcap('network_traffic.pcap', packets)
    print(f"Captured {PACKET_COUNT} packets and saved to 'network_traffic.pcap'")

# Extract features from captured packets and save to CSV
def extract_features():
    from scapy.all import rdpcap
    import pandas as pd

    print("Extracting features from captured packets...")

    packets = rdpcap('network_traffic.pcap')
    data = []

    for pkt in packets:
        if pkt.haslayer('IP'):
            data.append({
                'src_ip': pkt['IP'].src,
                'dst_ip': pkt['IP'].dst,
                'protocol': pkt['IP'].proto,
                'length': len(pkt)
            })

    df = pd.DataFrame(data)
    df['label'] = 0  # Normal traffic
    df.loc[df.sample(frac=0.1).index, 'label'] = 1  # Simulate 10% malicious traffic

    df['src_ip'] = df['src_ip'].astype('category').cat.codes
    df['dst_ip'] = df['dst_ip'].astype('category').cat.codes

    df.to_csv('network_data.csv', index=False)
    print("Extracted features and saved to 'network_data.csv'")

# Train a machine learning model using the extracted features
def train_model():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    import joblib

    print("Training the machine learning model...")

    df = pd.read_csv('network_data.csv')
    X = df[['src_ip', 'dst_ip', 'protocol', 'length']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, 'ids_model.joblib')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained. Accuracy: {accuracy * 100:.2f}%")

# Real-time intrusion detection using the trained model
def real_time_ids():
    from scapy.all import sniff
    import pandas as pd
    import joblib

    print("Starting real-time intrusion detection...")

    model = joblib.load('ids_model.joblib')

    def detect_intrusion(packet):
        if packet.haslayer('IP'):
            pkt_info = {
                'src_ip': packet['IP'].src,
                'dst_ip': packet['IP'].dst,
                'protocol': packet['IP'].proto,
                'length': len(packet)
            }
            pkt_df = pd.DataFrame([pkt_info])

            pkt_df['src_ip'] = pkt_df['src_ip'].astype('category').cat.codes
            pkt_df['dst_ip'] = pkt_df['dst_ip'].astype('category').cat.codes

            X_new = pkt_df[['src_ip', 'dst_ip', 'protocol', 'length']]

            prediction = model.predict(X_new)

            if prediction[0] == 1:
                print("🚨 Intrusion Detected!")
            else:
                print("✅ Normal Traffic")

    sniff(prn=detect_intrusion, store=0)

# Main function to run all tasks sequentially
def main():
    # Step 1: Install libraries
    install_libraries()

    # Step 2: Capture network packets
    capture_packets()

    # Step 3: Extract features from captured packets
    extract_features()

    # Step 4: Train the machine learning model
    train_model()

    # Step 5: Run real-time intrusion detection
    print("To run real-time intrusion detection, you may need elevated privileges.")
    real_time_ids()

if __name__ == '__main__':
    main()
