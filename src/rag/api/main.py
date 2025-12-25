from __future__ import annotations

from fastapi import FastAPI

from rag.api.routes import router


def create_app(test_mode: bool = False) -> FastAPI:
    """
    Фабрика FastAPI приложения.

    Parameters
    ----------
    test_mode : bool, optional
        Используется для отключения внешних зависимостей при тестах

    Returns
    -------
    FastAPI
        Инициализированное приложение с маршрутами
    """
    app = FastAPI(title="RAG RusLaw")
    app.include_router(router)
    if test_mode:
        app.state.test_mode = True
    return app
