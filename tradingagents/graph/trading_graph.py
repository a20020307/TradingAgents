import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.interface import set_config

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor

# 智能 LLM 路由工具，供分析师/agent 调用
def get_chain(prompt, tools, local_llm, cloud_llm):
    """
    智能路由LLM推理链：本地优先，用不了 bind_tools 自动用云端
    """
    try:
        if hasattr(local_llm, "bind_tools"):
            return prompt | local_llm.bind_tools(tools)
    except NotImplementedError:
        pass
    if hasattr(cloud_llm, "bind_tools"):
        return prompt | cloud_llm.bind_tools(tools)
    return prompt | cloud_llm

class TradingAgentsGraph:
    """主框架类"""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        set_config(self.config)
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # === 初始化 LLM（本地和云端）===
        # 本地 Ollama LLM
        self.ollama_llm = ChatOllama(
            model=self.config["deep_think_llm"],
            base_url=self.config["backend_url"]
        )
        # 云端 OpenAI LLM
        self.openai_llm = ChatOpenAI(
            model=self.config.get("openai_model", "gpt-3.5-turbo"),
            api_key=self.config["openai_api_key"],
            base_url=self.config.get("openai_base_url", "https://api.openai.com/v1")
        )

        # 为兼容原有接口
        self.deep_thinking_llm = self.ollama_llm
        self.quick_thinking_llm = self.openai_llm

        self.toolkit = Toolkit(config=self.config)

        # Memory
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        self.tool_nodes = self._create_tool_nodes()
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.toolkit,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}

        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        return {
            "market": ToolNode([
                self.toolkit.get_YFin_data_online,
                self.toolkit.get_stockstats_indicators_report_online,
                self.toolkit.get_YFin_data,
                self.toolkit.get_stockstats_indicators_report,
            ]),
            "social": ToolNode([
                self.toolkit.get_stock_news_openai,
                self.toolkit.get_reddit_stock_info,
            ]),
            "news": ToolNode([
                self.toolkit.get_global_news_openai,
                self.toolkit.get_google_news,
                self.toolkit.get_finnhub_news,
                self.toolkit.get_reddit_news,
            ]),
            "fundamentals": ToolNode([
                self.toolkit.get_fundamentals_openai,
                self.toolkit.get_finnhub_company_insider_sentiment,
                self.toolkit.get_finnhub_company_insider_transactions,
                self.toolkit.get_simfin_balance_sheet,
                self.toolkit.get_simfin_cashflow,
                self.toolkit.get_simfin_income_stmt,
            ]),
        }

    def propagate(self, company_name, trade_date):
        self.ticker = company_name
        init_agent_state = self.propagator.create_initial_state(company_name, trade_date)
        args = self.propagator.get_graph_args()
        if self.debug:
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)
            final_state = trace[-1]
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        self.curr_state = final_state
        self._log_state(trade_date, final_state)
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"]["current_response"],
                "judge_decision": final_state["investment_debate_state"]["judge_decision"],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json", "w"
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        self.reflector.reflect_bull_researcher(self.curr_state, returns_losses, self.bull_memory)
        self.reflector.reflect_bear_researcher(self.curr_state, returns_losses, self.bear_memory)
        self.reflector.reflect_trader(self.curr_state, returns_losses, self.trader_memory)
        self.reflector.reflect_invest_judge(self.curr_state, returns_losses, self.invest_judge_memory)
        self.reflector.reflect_risk_manager(self.curr_state, returns_losses, self.risk_manager_memory)

    def process_signal(self, full_signal):
        return self.signal_processor.process_signal(full_signal)
