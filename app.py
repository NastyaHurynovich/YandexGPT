import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import ChatYandexGPT
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

model_uri = "ds://bt12av2iqlohocbnvc8v"
default_prompt = "Твое имя Алекс, полное имя Зубик Александр Сергеевич. \nТы отвечаешь от лица мужского рода. \nТы робот. \nТы говоришь коротко и емко. \nТы был создан в Минске. \nТвое предназначение – обучать людей, отвечать на вопросы, помогать людям.\nТы эксперт в сфере машинного обучения. \nТы работаешь в Центре аналитических решений в ЗАО МТБанк.\nТы можешь консультировать по BigData, Data Lake, Машинному обучению и Искуственному интелекту."
yagpt_temperature = 0.5
yagpt_max_tokens = 5000


def main():
    st.title('YandexGPTchat (дообученная модель)')

    with st.sidebar:
        st.title('YandexGPT настройки')

    yagpt_api_key = st.sidebar.text_input("YaGPT API Key", type="password")
    if not yagpt_api_key:
        st.info(
            "Укажите YandexGPT API ключ для запуска чат-бота")
        st.stop()

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Привет! Как я могу вам помочь?")
    view_messages = st.expander("Просмотр истории сообщений")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", default_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    model = ChatYandexGPT(api_key=yagpt_api_key, model_uri=model_uri, temperature=yagpt_temperature,
                          max_tokens=yagpt_max_tokens)

    chain = prompt | model
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="history",
    )

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Если пользователь вводит новое приглашение, сгенерировать и отобразить новый ответ
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        # Примечание: новые сообщения автоматически сохраняются в историю по длинной цепочке во время запуска
        config = {"configurable": {"session_id": "any"}}
        response = chain_with_history.invoke({"question": prompt}, config)
        st.chat_message("ai").write(response.content)

    # Отобразить сообщения в конце, чтобы вновь сгенерированные отображались сразу
    with view_messages:
        """
        История сообщений, инициализированная с помощью:
        ```python
        msgs = StreamlitChatMessageHistory(key="langchain_messages")
        ```

        Содержание `st.session_state.langchain_messages`:
        """
        view_messages.json(st.session_state.langchain_messages)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.write(f"Что-то пошло не так. {str(e)}")
