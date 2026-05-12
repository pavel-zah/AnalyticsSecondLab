import sys
import pandas as pd
from src.agent.agent import build_agent
from langchain_core.messages import HumanMessage
import json


def analyze_review(review_text: str) -> dict:
    agent = build_agent()

    try:
        input_data = {
            "messages": [HumanMessage(content=review_text)]
        }

        agent_response = agent.invoke(
            input=input_data,
            config=None
        )

        response = {
            "review_sentiment": agent_response["structured_response"].review_sentiment,
            "review_topic": agent_response["structured_response"].review_topic,
            "review_text": review_text
        }
        return response

    except Exception:
        return {
            "error": "ошибка при обращении к llm",
            "review_text": review_text
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python script.py <имя_csv_файла>")
        sys.exit(1)

    csv_file = sys.argv[1]

    df = pd.read_csv(csv_file, sep=';')
    output_results = []

    for _, row in df.iterrows():
        review_text = row.review_text
        review_analysis = analyze_review(review_text)
        output_results.append(review_analysis)

    output_file = "analysis_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=4)

    print(f"Результаты сохранены в {output_file}")