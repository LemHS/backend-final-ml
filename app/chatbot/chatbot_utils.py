import pandas as pd
import os
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from rapidfuzz.distance import JaroWinkler


def ReciprocalRankFusion(rank_df, k, name):
    rank_df[name] = rank_df.apply(lambda x: sum(1 / (k + x[col]) if "jaro_winkler" not in col else 2 / (k + x[col]) for col in rank_df.columns if "rank" in col), axis=1).rank(ascending=False, method="min")
    return rank_df

class JaroWinklerRanking():
    def __init__(self, doc_df):
        self.doc_df = doc_df
        self.doc_len = len(doc_df)
        self.df = pd.DataFrame({"id": [i for i in range(self.doc_len)]})

    def rank(self, query_dict):
        for query_type, query in query_dict.items():
            if query_type in ["Nama Obat", "Manufaktur"]:
                if isinstance(query, list):
                    for i, item in enumerate(query):
                        results = self.doc_df[[query_type]].copy()
                        results["score"] = results[query_type].apply(lambda x: JaroWinkler.similarity(x, item))
                        results_id = results.sort_values(by="score", ascending=False).index.to_list()
                        results_score = results.sort_values(by="score", ascending=False)["score"].to_list()

                        col_df = pd.DataFrame({"id": results_id, f"{query_type}_{i}_score": results_score})
                        col_df["id"] = col_df["id"].astype(int)
                        self.df = pd.merge(left=self.df, right=col_df, how="left", on="id")
                        self.df[f"{query_type}_{i}_score"] = self.df[f"{query_type}_{i}_score"].fillna(self.df[f"{query_type}_{i}_score"].min())
                else:
                    results = self.doc_df[[query_type]].copy()
                    results["score"] = results[query_type].apply(lambda x: JaroWinkler.similarity(x, query))
                    results_id = results.sort_values(by="score", ascending=False).index.to_list()
                    results_score = results.sort_values(by="score", ascending=False)["score"].to_list()

                    col_df = pd.DataFrame({"id": results_id, f"{query_type}_score": results_score})
                    col_df["id"] = col_df["id"].astype(int)
                    self.df = pd.merge(left=self.df, right=col_df, how="left", on="id")
                    self.df[f"{query_type}_score"] = self.df[f"{query_type}_score"].fillna(self.df[f"{query_type}_score"].min())

        rank_df = self.df.copy()
        rank_df["jaro_winkler_rank"] = rank_df.apply(lambda x: sum(x[col] for col in rank_df.columns if "score" in col) / len(query_dict), axis=1).rank(ascending=False, method="min")

        return rank_df[["id", "jaro_winkler_rank"]]        


class LexicalRanking():
    def __init__(self, retrievers, doc_df):
        self.retrievers = retrievers
        self.doc_df = doc_df
        self.doc_len = len(doc_df)
        self.df = pd.DataFrame({"id": [i for i in range(self.doc_len)]})

    def rank(self, query_dict):
        for query_type, query in query_dict.items():
            if query_type not in ["Nama Obat", "Manufaktur"]:
                if isinstance(query, list):
                    for i, item in enumerate(query):
                        results = self.retrievers[query_type].invoke(item)
                        results_id = [result.metadata["doc_id"] for result in results]
                        results_rank = [i for i in range(1, self.doc_len + 1)]

                        col_df = pd.DataFrame({"id": results_id, f"{query_type}_{i}_rank": results_rank})
                        col_df["id"] = col_df["id"].astype(int)
                        self.df = pd.merge(left=self.df, right=col_df, how="left", on="id")
                else:
                    results = self.retrievers[query_type].invoke(query)
                    results_id = [result.metadata["doc_id"] for result in results]
                    results_rank = [i for i in range(1, self.doc_len + 1)]

                    col_df = pd.DataFrame({"id": results_id, f"{query_type}_rank": results_rank})
                    col_df["id"] = col_df["id"].astype(int)
                    self.df = pd.merge(left=self.df, right=col_df, how="left", on="id")

        rank_df = ReciprocalRankFusion(self.df, 60, "lexical_rank")

        return rank_df[["id", "lexical_rank"]]
    
class SemanticRanking():
    def __init__(self, retriever, doc_df):
        self.retriever = retriever
        self.doc_df = doc_df
        self.doc_len = len(doc_df)
        self.df = pd.DataFrame({"id": [i for i in range(self.doc_len)]})

    def rank(self, query_dict):
        for query_type, query in query_dict.items():
            if query_type not in ["Nama Obat", "Manufaktur"]:
                if isinstance(query, list):
                    for i, item in enumerate(query):
                        results = self.retriever.similarity_search_with_relevance_scores(item, filter={"column": query_type}, k=self.doc_len)
                        results_id = [result[0].metadata["doc_id"] for result in results]
                        results_score = [result[1] for result in results]

                        col_df = pd.DataFrame({"id": results_id, f"{query_type}_{i}_score": results_score})
                        col_df["id"] = col_df["id"].astype(int)
                        self.df = pd.merge(left=self.df, right=col_df, how="left", on="id")
                        self.df[f"{query_type}_{i}_score"] = self.df[f"{query_type}_{i}_score"].fillna(self.df[f"{query_type}_{i}_score"].min())
                else:
                    results = self.retriever.similarity_search_with_relevance_scores(query, filter={"column": query_type}, k=self.doc_len)
                    results_id = [result[0].metadata["doc_id"] for result in results]
                    results_score = [result[1] for result in results]

                    col_df = pd.DataFrame({"id": results_id, f"{query_type}_score": results_score})
                    col_df["id"] = col_df["id"].astype(int)
                    self.df = pd.merge(left=self.df, right=col_df, how="left", on="id")
                    self.df[f"{query_type}_score"] = self.df[f"{query_type}_score"].fillna(self.df[f"{query_type}_score"].min())

        rank_df = self.df.copy()
        rank_df["semantic_rank"] = rank_df.apply(lambda x: sum(x[col] for col in rank_df.columns if "score" in col) / len(query_dict), axis=1).rank(ascending=False, method="min")

        return rank_df[["id", "semantic_rank"]]
    
class CreateRetriever():
    def __init__(self, df, col_to_embed):
        self.df = df
        self.col_to_embed = col_to_embed

    def create_semantic_retriever(self, chroma_path, embedding_model):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': False}
        embed_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        if os.path.isdir(chroma_path):
            vector_db = Chroma(
                collection_name="halodoc_embeddings",
                embedding_function=embed_model,
                persist_directory=chroma_path,  # Where to save data locally, remove if not necessary
            )
        else:
            texts = []
            metadatas = []
            ids = []

            for _, row in self.df.iterrows():
                doc_id = str(row.name)  # or another unique key
                
                for col in self.col_to_embed:
                    text = str(row[col])
                    texts.append(text)
                    metadatas.append({"doc_id": doc_id, "column": col})
                    ids.append(f"{doc_id}_{col}")

            vector_db = Chroma.from_texts(
                texts=texts,
                embedding=embed_model,
                metadatas=metadatas,
                ids=ids,
                collection_name="halodoc_embeddings",
                persist_directory=chroma_path
            )

        return vector_db
    
    def create_lexical_retriever(self):
        texts = []
        metadatas = []
        ids = []

        lexical_retrievers = {col:"" for col in self.col_to_embed}

        for col in self.col_to_embed:
            
            for _, row in self.df.iterrows():
                doc_id = str(row.name)  # or another unique key
                text = str(row[col])
                texts.append(text)
                metadatas.append({"doc_id": doc_id, "column": col})
                ids.append(f"{doc_id}_{col}")

            lexical_retrievers[col] = BM25Retriever.from_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
            )

            lexical_retrievers[col].k = len(self.df)

            texts = []
            metadatas = []
            ids = []

        return lexical_retrievers