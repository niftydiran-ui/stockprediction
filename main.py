import argparse
from agents.agent import Agent
from rich import print as rprint

def parse_args():
    p = argparse.ArgumentParser(description="Agentic Stock Predictor")
    p.add_argument("--tickers", nargs="+", required=True, help="Tickers, e.g., AAPL MSFT")
    p.add_argument("--period", default="5y", help="History window for yfinance (e.g., 2y, 5y, max)")
    p.add_argument("--model", choices=["xgb","linear"], default="xgb")
    p.add_argument("--threshold", type=float, default=0.55, help="Go long if P(up) >= threshold")
    p.add_argument("--out", default="artifacts", help="Output directory")
    return p.parse_args()

def main():
    args = parse_args()
    agent = Agent(model_name=args.model, threshold=args.threshold, out_dir=args.out)
    for t in args.tickers:
        rprint(f"[bold cyan]\n=== Running pipeline for {t} ===[/]")
        report = agent.run(ticker=t, period=args.period)
        rprint(f"[green]Summary for {t}:[/] {report['metrics']}")
        rprint(f"[dim]Artifacts saved to: {report['artifact_dir']}[/]")

if __name__ == "__main__":
    main()
