from opensearchpy import OpenSearch
import langchain as langchain
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import ChatYandexGPT
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts.prompt import PromptTemplate
from langchain.vectorstores import OpenSearchVectorSearch
from yandex_chain import YandexEmbeddings

#model_uri = "ds://bt12av2iqlohocbnvc8v"
model_uri = 'gpt://b1g9kncn7jjlvp0q4vtq/yandexgpt/latest'
default_prompt = """
    Представь себе, что ты сотрудник Yandex Cloud. Твоя задача - вежливо и по мере своих сил отвечать на все вопросы собеседника.
    Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста. Если ответа в тексте нет, напиши "Я не знаю".
    Текст:
    -----
    {context}
    -----
    Вопрос:
    {query}
"""

yagpt_temperature = 0
yagpt_max_tokens = 8000
HOSTS = ['rc1a-mv9c6qsp22vk2euf.mdb.yandexcloud.net', 'rc1a-r5eqg5vkva2117a8.mdb.yandexcloud.net']
PASS = 'zxcdewqas'
CA = './root.crt'

def main():
    st.title('YandexGPTchat (with RAG)')

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


    folder_id = 'b1g9kncn7jjlvp0q4vtq'
    token = 't1.9euelZqTmsiMzMiSypiLk8jJjJ3Ixu3rnpWay8qUi46OzMuOz8zIiouTzpjl8_cEWgRL-e9RG2pD_d3z90QIAkv571EbakP9zef1656VmpSRnJ3JzJLKksqQmp2Xm8ua7_zN5_XrnpWazIyciYrIlIuWns2Qx5jLiZjv_cXrnpWalJGcncnMksqSypCanZeby5o.DMzOsbWpbwbpjpofukr_igr6eDjlYS5l5n6xyB6-wgp2DBg6ZR68JrZKhwCEHXMQY6qUcGJcW20qNSCsNivDBw'
    embeddings = YandexEmbeddings(folder_id=folder_id, iam_token=token)
    embeddings.sleep_interval = 0.02

    docsearch = OpenSearchVectorSearch(
        embedding_function=embeddings,
        index_name="dz-test-1000-200",
        opensearch_url=HOSTS,
        http_auth=("admin", PASS),
        use_ssl=True,
        verify_certs=True,
        ca_certs=CA,
        engine="lucene",
    )

    model = ChatYandexGPT(api_key=yagpt_api_key, model_uri=model_uri, temperature=yagpt_temperature,
                          max_tokens=yagpt_max_tokens)

    # Промпт для обработки документов
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )

    # Промпт для языковой модели
    document_variable_name = "context"
    stuff_prompt_override = """
        Представь себе, что ты сотрудник Yandex Cloud. Твоя задача - вежливо и по мере своих сил отвечать на все вопросы собеседника.
        Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста. Если ответа в тексте нет, напиши "Я не знаю".
        Текст:
        -----
        {context}
        -----
        Вопрос:
        {query}
    """
    prompt = PromptTemplate(
        template=stuff_prompt_override, input_variables=["context", "query"]
    )

    # Создаём цепочку
    llm_chain = langchain.chains.LLMChain(llm=model, prompt=prompt)
    chain = langchain.chains.combine_documents.stuff.StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="query",
        history_messages_key="history",
    )

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Если пользователь вводит новое приглашение, сгенерировать и отобразить новый ответ
    if query := st.chat_input():
        st.chat_message("human").write(query)
        # # Примечание: новые сообщения автоматически сохраняются в историю по длинной цепочке во время запуска
        config = {"configurable": {"session_id": "any"}}
        # # response = chain_with_history.invoke({"question": prompt}, config)
        response = chain_with_history.invoke({"query": query, "input_documents":  docsearch.similarity_search(query, k=2)}, config)
        st.chat_message("ai").write(response['output_text'])

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
