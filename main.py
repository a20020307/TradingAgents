import yaml
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.agents.binance_agent import BinanceTradingAgent

# 加载YAML配置
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 初始化Binance Agent
binance_cfg = config.get("binance", {})
binance_agent = BinanceTradingAgent(
    api_key=binance_cfg["api_key"],
    api_secret=binance_cfg["api_secret"],
    symbol=binance_cfg.get("symbol", "BTCUSDT"),
    testnet=binance_cfg.get("testnet", True)
)

# 初始化TradingAgentsGraph
ta = TradingAgentsGraph(
    debug=True,
    config=config,
)

# 推理决策（用BTCUSDT和今天日期）
from datetime import date
today = date.today().isoformat()

_, decision = ta.propagate("BTCUSDT", today)
print("决策结果：", decision)

