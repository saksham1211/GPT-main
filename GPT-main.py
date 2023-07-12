#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time

## author: Saksham Dubey
'''Added memory to the model, to remeber the conversation which can we reused
to generate the outputs.'''

from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss

from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.llms import OpenAI


# memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
# memory.save_context({"input": "hi"}, {"output": "whats up"})



# history = ChatMessageHistory()
# history.add_user_message("hi")
# history.add_ai_message("hi there!")


# embedding_size = 512   
# index = faiss.IndexFlatL2(embedding_size)
# persist_directory = os.environ.get('PERSIST_DIRECTORY')
# source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')

# embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")


# # embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
# embedding_fn = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
# vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

# retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
# memory = VectorStoreRetrieverMemory(retriever=retriever)



load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

# _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

# Relevant pieces of previous conversation:
# {history}

# (You do not need to use these pieces of information if not relevant)

# Current conversation:
# Human: {input}
# AI:"""
# PROMPT = PromptTemplate(
#     input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
# )

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # memeory=VectorStoreRetrieverMemory(retriever=retriever)
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False, n_gpu_layers=10, n_threads=16)
            # memory=ConversationSummaryMemory(llm=llm)
            
            # memory = ConversationSummaryMemory.from_messages(llm=llm, chat_memory=history, return_messages=True)
            # memory.load_memory_variables({})
        case "GPT4All":
            llm = GPT4All(model=model_path,n_predict=1024,  n_threads=16, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
    
    qa = RetrievalQA.from_chain_type(llm=llm,  chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        print(answer)
        print(res['result'])
        end = time.time()
        # Print the result
        print("\n\n> Question:")
        print(query)
        # history.add_message(query)
        # history.add_ai_message(res['result'])
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

       
        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
