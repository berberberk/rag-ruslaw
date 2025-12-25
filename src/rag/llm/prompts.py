from __future__ import annotations

FALLBACK_TEXT = "Недостаточно информации в предоставленном контексте."


def build_prompt(question: str, context: str) -> list[dict]:
    """
    Формирует список сообщений для OpenRouter.

    Parameters
    ----------
    question : str
        Пользовательский вопрос
    context : str
        Подготовленный контекст (номерованные чанки)

    Returns
    -------
    list[dict]
        Сообщения для чат-комплишена
    """
    system = (
        "Ты — юридический ассистент. Отвечай только на основе CONTEXT. "
        "Если информации недостаточно, ответь: 'Недостаточно информации в предоставленном контексте.' "
        "Обязательно указывай цитаты (источники) как список doc_id/title/docdate."
    )
    user = f"Question: {question}\nCONTEXT:\n{context}\nOutput:\nAnswer (markdown) + Citations."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
