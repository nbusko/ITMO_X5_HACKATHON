import os
import json

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from fastapi import status
import faiss
import numpy as np
import pandas as pd
import sqlalchemy as sa
import aiohttp

from src.pg_models import IndexMetainfo, Texts
import src.config as config
# from pg_models import IndexMetainfo, Texts
# import config as config


async def get_text_embedding(text):
    """
    Retrieves the embedding vector for a given
    text by sending a GET request to an embedding service API.

    Args:
        text (str): The text for which the
        embedding will be retrieved.

    Returns:
        list: The embedding vector of the query text.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{os.getenv('EMBEDDER_URL')}/search?text={text}"
        ) as resp:
            response = await resp.json()

    return response["query_embedding"]


async def get_llm_answer(history):
    """ """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{os.getenv('NEURAL_URL')}/generate",
            json={
                "chat_history": history,
                "max_new_tokens": config.max_new_tokens,
            },
        ) as resp:
            response = await resp.json()

    try:
        answer = json.loads(response["answer"])["answer"]
    except:
        # if LLM failed to generate json
        answer = None

    return answer


async def rewrite_query(chat_history):
    """ """

    lines = "\n".join(
        [
            f"{'ПОЛЬЗОВАТЕЛЬ ' if el.role == 'user' else 'АССИСТЕНТ '}: {el.content}"
            for el in chat_history
        ]
    )
    history = [{"role": "user", "content": config.qr_prompt.format(history=lines)}]

    llm_answer = await get_llm_answer(history)

    if llm_answer:
        return llm_answer

    return chat_history[0].content


async def check_popularity(last_query : str) -> str:
    """ """
    history = [{"role": "user", "content": config.popularity_prompt.format(query=last_query)}]

    llm_answer = await get_llm_answer(history)

    if llm_answer:
        return llm_answer
    
    return "not_domain"

async def check_domain(full_query : str) -> str:
    """ """
    history = [{"role": "user", "content": config.domain_prompt.format(query=full_query)}]

    llm_answer = await get_llm_answer(history)

    if llm_answer:
        return llm_answer
    
    return "not_domain"

async def qa_stuff(query, passages):
    """ """

    lines = "\n".join(
        [
            f"=== Фрагмент {idx + 1} ===\n{config.faiss_func(el)}"
            for idx, el in enumerate(passages)
        ]
    )
    history = [
        {
            "role": "user",
            "content": config.stuff_prompt.format(query=query, passages=lines),
        }
    ]

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{os.getenv('NEURAL_URL')}/generate",
            json={
                "chat_history": history,
                "max_new_tokens": config.max_new_tokens,
            },
        ) as resp:
            response = await resp.json()

    return json.loads(response["answer"])["answer"]


def faiss_search_result(query_embedding, f_index: faiss.IndexFlatL2):
    """
    Find the nearest neighbor embedding IDs and
    distances for a given query embedding using a Faiss index.
    """
    embedding_distances, embedding_ids = f_index.search(
        np.array([query_embedding]).astype("float32"), config.topn
    )

    return embedding_ids[0], embedding_distances[0]


async def insert_data_to_pg(cols, session: AsyncSession):
    """
    Insert data into a PostgreSQL database
    table using SQLAlchemy and return the primary key of the inserted row.
    """
    q = sa.insert(IndexMetainfo).values(cols).returning(IndexMetainfo.index_metainf_id)
    q = await session.execute(q)
    p_id = q.scalar_one_or_none()

    return p_id


async def insert_text(idx_id, p_id, session: AsyncSession):
    q = sa.insert(Texts).values(index_id=idx_id, metainf_id=p_id)
    await session.execute(q)


async def insert_data(
    row: dict,
    faiss_str: str,
    session: AsyncSession,
    f_index: faiss.IndexFlatL2,
):
    """
    Insert data into a PostgreSQL database table and update a Faiss index.
    """
    p_id = await insert_data_to_pg(cols=row, session=session)

    n_t = f_index.ntotal
    e = await get_text_embedding(faiss_str)
    embedding = np.array([e]).astype("float32")
    f_index.add(embedding)

    await insert_text(idx_id=n_t, p_id=p_id, session=session)


async def acc_testing(session: AsyncSession, f_index: faiss.IndexFlatL2):
    db_df = pd.read_csv(config.path_to_testfile)

    top_3, top_5, top_10 = 0, 0, 0
    for idx, row in db_df.iterrows():
        q_emb = await qa_stuff(row["Характеристика товара"])

        res = await search_by_embedding(q_emb, session, f_index)
        res = [el["index_metainf_id"] for el in res]

        if row["Артикул"] in res[:3]:
            top_3 += 1

        if row["Артикул"] in res[:5]:
            top_5 += 1

        if row["Артикул"] in res:
            top_10 += 1

    print(f"top_10 = {top_10} --- top_5 = {top_5} --- IDX = {idx}", flush=True)
    print(
        f"acc@10 = {top_10 / idx} --- acc@5 = {top_5 / idx} --- acc@3 = {top_3 / idx}",
        flush=True,
    )


async def init_db(session: AsyncSession, f_index: faiss.IndexFlatL2):
    """
    Initializes the database by inserting data from CSV files
    into corresponding database tables and updating Faiss index.
    """
    if f_index.ntotal:
        return

    db_df = pd.read_csv(config.path_to_excel)
    # db_df = db_df.head(5)

    if db_df.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File is broken or wrong columns are specified",
        )

    for _, row in db_df.iterrows():
        # TODO
        # look at abbr_decoding in config

        row = {k: v if not pd.isna(v) else None for k, v in row.to_dict().items()}

        await insert_data(
            row=row,
            faiss_str=config.faiss_func(row),
            session=session,
            f_index=f_index,
        )

    faiss.write_index(f_index, config.path_to_index)
    f_index = faiss.read_index(config.path_to_index)

    await session.commit()


async def get_metainf_by_text(faiss_id, session: AsyncSession):
    """
    Retrieves metadata information from a
    database table based on a given ID.
    """
    q = (
        sa.select()
        .with_only_columns(Texts.metainf_id)
        .where(Texts.index_id == faiss_id)
    )
    q = await session.execute(q)
    res = q.fetchone()
    p_id = res.metainf_id

    q = sa.select(IndexMetainfo).where(IndexMetainfo.index_metainf_id == p_id)
    q = await session.execute(q)
    res = q.fetchone()[0]  # why [0]?

    return {c.name: str(getattr(res, c.name)) for c in res.__table__.columns}


async def search_by_embedding(
    embedding, session: AsyncSession, f_index: faiss.IndexFlatL2
):
    """
    Retrieve metadata information from a
    database table based on a given embedding.
    """
    emb_ids, emb_dist = faiss_search_result(embedding, f_index)

    res_dict = []
    for index_id, dist in zip(emb_ids, emb_dist):
        metainfo = await get_metainf_by_text(index_id, session)
        if not metainfo:
            continue

        metainfo["distance"] = str(dist)

        res_dict.append(metainfo)

    return res_dict


# TODO
# async def update(
#     data_to_add,
#     type,
#     session: AsyncSession,
#     f_indexes: dict[str, tuple[faiss.IndexFlatL2, str]],
# ):
#     """
#     Insert data into a PostgreSQL database table
#     and update a Faiss index.

#     Args:
#         data_to_add: A dictionary representing the
#         data to be inserted into the database table.

#         type: A string representing the type of data (resumes or vacancies).

#         session: An AsyncSession object representing the database session.

#         f_indexes: A dictionary containing Faiss indexes
#         for different types of data.

#     Returns:
#         None
#     """
#     _cls, faiss_str_func = TYPE_TABLE[type]

#     await insert_data(
#         row=data_to_add,
#         faiss_str=faiss_str_func(data_to_add),
#         table=_cls,
#         session=session,
#         f_index=f_indexes[type][0],
#     )

#     for _, v in f_indexes.items():
#         faiss.write_index(v[0], v[1])
#         v[0] = faiss.read_index(v[1])

#     await session.commit()
