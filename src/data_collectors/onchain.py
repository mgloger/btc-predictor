import requests
import pandas as pd


class OnChainCollector:
    """Collects blockchain-level data from Glassnode / CryptoQuant."""

    def __init__(self, config):
        self.glassnode_key = config["GLASSNODE_API_KEY"]
        self.base_url = "https://api.glassnode.com/v1/metrics"

    def _fetch(self, endpoint: str, asset="BTC", interval="24h", days=365) -> pd.DataFrame:
        params = {
            "a": asset,
            "i": interval,
            "api_key": self.glassnode_key,
        }
        resp = requests.get(f"{self.base_url}/{endpoint}", params=params)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        df["t"] = pd.to_datetime(df["t"], unit="s")
        df.set_index("t", inplace=True)
        return df

    def get_active_addresses(self, days=365) -> pd.DataFrame:
        """Number of unique active addresses (network health)."""
        return self._fetch("addresses/active_count", days=days)

    def get_exchange_netflow(self, days=365) -> pd.DataFrame:
        """Net BTC flowing in/out of exchanges (sell pressure proxy)."""
        return self._fetch("transactions/transfers_volume_exchanges_net", days=days)

    def get_mvrv_ratio(self, days=365) -> pd.DataFrame:
        """Market Value to Realized Value — key over/undervaluation metric."""
        return self._fetch("market/mvrv", days=days)

    def get_hodl_waves(self, days=365) -> pd.DataFrame:
        """HODL waves — age distribution of UTXOs."""
        return self._fetch("supply/hodl_waves", days=days)

    def get_hash_rate(self, days=365) -> pd.DataFrame:
        """Network hash rate — miner confidence indicator."""
        return self._fetch("mining/hash_rate_mean", days=days)

    def get_sopr(self, days=365) -> pd.DataFrame:
        """Spent Output Profit Ratio — profit-taking indicator."""
        return self._fetch("indicators/sopr", days=days)