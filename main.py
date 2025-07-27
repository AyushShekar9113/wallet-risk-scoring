import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

# -----------------------------
# Step 1: Load Wallet Addresses
# -----------------------------
wallets_df = pd.read_csv("Wallet id - Sheet1.csv")
wallet_addresses = wallets_df["wallet_id"].str.lower().tolist()

# -----------------------------
# Step 2: Define The Graph Query
# -----------------------------
GRAPH_API_URL = "https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2"

def query_compound(wallet):
    query = """
    query ($user: String!) {
        account(id: $user) {
        tokens {
            symbol
            supplyBalanceUnderlying
            borrowBalanceUnderlying
        }
    }
    }
    """
    response = requests.post(
        GRAPH_API_URL,
        json={"query": query, "variables": {"user": wallet}}
    )
    if response.status_code == 200:
        return response.json()
    else:
        return None

# -----------------------------
# Step 3: Extract Transaction Features
# -----------------------------
def extract_wallet_metrics(wallet):
    data = query_compound(wallet)
    total_supply, total_borrow = 0, 0

    if data and data.get("data") and data["data"].get("account"):
        tokens = data["data"]["account"]["tokens"]
        for token in tokens:
            total_supply += float(token.get("supplyBalanceUnderlying", 0))
            total_borrow += float(token.get("borrowBalanceUnderlying", 0))
    
    net_position = total_supply - total_borrow
    borrow_supply_ratio = total_borrow / total_supply if total_supply > 0 else 2.0  # Cap as risky if no supply
    
    return {
        "wallet_id": wallet,
        "total_supply": total_supply,
        "total_borrow": total_borrow,
        "net_position": net_position,
        "borrow_supply_ratio": min(borrow_supply_ratio, 2.0)  # cap for extreme cases
    }

# -----------------------------
# Step 4: Batch Process Wallets
# -----------------------------
wallet_data = []

for i, wallet in enumerate(wallet_addresses):
    print(f"[{i+1}/{len(wallet_addresses)}] Processing {wallet}...")
    try:
        result = extract_wallet_metrics(wallet)
        wallet_data.append(result)
        time.sleep(0.3)  # Rate limiting for Graph API
    except Exception as e:
        print(f"Error with wallet {wallet}: {e}")
        continue

df = pd.DataFrame(wallet_data)

# -----------------------------
# Step 5: Normalize and Score
# -----------------------------
features = df[['total_supply', 'total_borrow', 'net_position', 'borrow_supply_ratio']]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

# Weighted scoring
# Lower borrow_supply_ratio = lower risk
df['score'] = (
    (1 - scaled[:, 3]) * 0.4 +  # Borrow/Supply ratio (inverse)
    scaled[:, 0] * 0.2 +        # Total supply
    scaled[:, 2] * 0.3 +        # Net position
    (1 - scaled[:, 1]) * 0.1    # Total borrow (inverse)
) * 1000

df['score'] = df['score'].astype(int)

# -----------------------------
# Step 6: Export to CSV
# -----------------------------
output_df = df[['wallet_id', 'score']]
output_path = "wallet_scores.csv"  # OR an absolute path if you want
output_df.to_csv(output_path, index=False)
output_path
