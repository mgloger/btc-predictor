import requests
import pandas as pd
from datetime import datetime, timedelta


class OnChainCollector:
    """
    Collects on-chain Bitcoin data using FREE APIs only.
    No Glassnode required.
    
    Sources:
        - mempool.space (no key needed)
        - blockchain.com (no key needed)
        - blockchair.com (no key needed, rate limited)
    """

    # ─── Mempool.space (completely free, no key) ───

    def get_hash_rate(self, timeframe="1y") -> pd.DataFrame:
        """Network hash rate from mempool.space."""
        resp = requests.get(
            f"https://mempool.space/api/v1/mining/hashrate/{timeframe}"
        )
        resp.raise_for_status()
        data = resp.json()["hashrates"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        df.rename(columns={"avgHashrate": "hash_rate"}, inplace=True)
        return df[["hash_rate"]]

    def get_difficulty(self, timeframe="1y") -> pd.DataFrame:
        """Mining difficulty adjustments from mempool.space."""
        resp = requests.get(
            f"https://mempool.space/api/v1/mining/hashrate/{timeframe}"
        )
        resp.raise_for_status()
        data = resp.json()["difficulty"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        return df[["difficulty"]]

    def get_mempool_stats(self) -> dict:
        """Current mempool statistics (tx count, fee rates, size)."""
        resp = requests.get("https://mempool.space/api/mempool")
        resp.raise_for_status()
        return resp.json()

    def get_recent_blocks(self, count=10) -> list[dict]:
        """Recent block data (size, weight, tx count, fees)."""
        resp = requests.get(f"https://mempool.space/api/v1/blocks")
        resp.raise_for_status()
        return resp.json()[:count]

    # ─── Blockchain.com (completely free, no key) ───

    def get_network_stats(self) -> dict:
        """
        Aggregate blockchain stats from blockchain.com:
        - market_price_usd
        - hash_rate (GH/s)
        - total_fees_btc
        - n_tx (daily transactions)
        - n_blocks_mined
        - minutes_between_blocks
        - trade_volume_usd
        """
        resp = requests.get("https://api.blockchain.info/stats")
        resp.raise_for_status()
        return resp.json()

    def get_chart_data(self, chart_name: str, days: int = 365) -> pd.DataFrame:
        """
        Fetch time-series chart data from blockchain.com.
        
        Available charts:
            - total-bitcoins        : Total BTC in circulation
            - market-price          : Market price (USD)
            - market-cap            : Market capitalization
            - trade-volume          : Exchange trade volume (USD)
            - transactions-per-second
            - mempool-size          : Mempool size (bytes)
            - mempool-count         : Mempool tx count
            - miners-revenue        : Total miner revenue (USD)
            - hash-rate             : Hash rate (TH/s)
            - difficulty            : Mining difficulty
            - estimated-transaction-volume-usd
            - n-unique-addresses    : Unique active addresses
            - n-transactions        : Daily transaction count
            - output-volume         : Total output volume (BTC)
        """
        timespan = f"{days}days"
        resp = requests.get(
            f"https://api.blockchain.info/charts/{chart_name}",
            params={"timespan": timespan, "format": "json", "rollingAverage": "8hours"},
        )
        resp.raise_for_status()
        data = resp.json()["values"]
        df = pd.DataFrame(data)
        df["x"] = pd.to_datetime(df["x"], unit="s")
        df.set_index("x", inplace=True)
        df.rename(columns={"y": chart_name}, inplace=True)
        return df

    # ─── Convenience methods using blockchain.com charts ───

    def get_active_addresses(self, days=365) -> pd.DataFrame:
        """Unique addresses active daily."""
        return self.get_chart_data("n-unique-addresses", days)

    def get_transaction_count(self, days=365) -> pd.DataFrame:
        """Daily confirmed transactions."""
        return self.get_chart_data("n-transactions", days)

    def get_miner_revenue(self, days=365) -> pd.DataFrame:
        """Total daily miner revenue in USD (block reward + fees)."""
        return self.get_chart_data("miners-revenue", days)

    def get_transaction_volume(self, days=365) -> pd.DataFrame:
        """Estimated daily transaction volume in USD."""
        return self.get_chart_data("estimated-transaction-volume-usd", days)

    def get_mempool_count(self, days=365) -> pd.DataFrame:
        """Number of unconfirmed transactions in mempool over time."""
        return self.get_chart_data("mempool-count", days)

    def get_total_supply(self, days=365) -> pd.DataFrame:
        """Total Bitcoin in circulation."""
        return self.get_chart_data("total-bitcoins", days)

    def get_market_cap(self, days=365) -> pd.DataFrame:
        """Market capitalization over time."""
        return self.get_chart_data("market-cap", days)

    # ─── Blockchair (free, rate limited to ~30 req/min) ───

    def get_blockchair_stats(self) -> dict:
        """
        Comprehensive Bitcoin network stats from Blockchair:
        - blocks, transactions, outputs
        - circulation, blockchain_size
        - average_transaction_fee_usd
        - mempool_transactions, mempool_size
        - suggested_transaction_fee_per_byte
        """
        resp = requests.get("https://api.blockchair.com/bitcoin/stats")
        resp.raise_for_status()
        return resp.json()["data"]