import requests
import pandas as pd


class MacroDataCollector:
    """Collects macroeconomic data from FRED and other sources."""

    def __init__(self, config):
        self.fred_key = config["FRED_API_KEY"]
        self.fred_base = "https://api.stlouisfed.org/fred/series/observations"

    def _fetch_fred(self, series_id: str) -> pd.DataFrame:
        params = {
            "series_id": series_id,
            "api_key": self.fred_key,
            "file_type": "json",
        }
        resp = requests.get(self.fred_base, params=params)
        resp.raise_for_status()
        data = resp.json()["observations"]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.set_index("date", inplace=True)
        return df[["value"]].rename(columns={"value": series_id})

    def get_fed_funds_rate(self) -> pd.DataFrame:
        return self._fetch_fred("FEDFUNDS")

    def get_cpi(self) -> pd.DataFrame:
        """Consumer Price Index — inflation proxy."""
        return self._fetch_fred("CPIAUCSL")

    def get_dxy(self) -> pd.DataFrame:
        """US Dollar Index — dollar strength."""
        return self._fetch_fred("DTWEXBGS")

    def get_m2_money_supply(self) -> pd.DataFrame:
        """M2 money supply — liquidity measure."""
        return self._fetch_fred("M2SL")

    def get_sp500(self) -> pd.DataFrame:
        """S&P 500 — risk-on/risk-off proxy."""
        return self._fetch_fred("SP500")

    def get_treasury_10y(self) -> pd.DataFrame:
        """10-Year Treasury Yield."""
        return self._fetch_fred("DGS10")