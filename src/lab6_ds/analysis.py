"""Testable extensions used by exercises 8–10 of the existing notebook."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

MODEL_ID = "pysentimiento/robertuito-sentiment-analysis"
MODEL_REVISION = "a2cc0f67ebd705c55191e25a05ba23d885fcc09b"
LABELS = ["NEG", "NEU", "POS"]


def centrality_tables(graph, comments):
    """Use unweighted paths: comment frequency is strength, not distance."""
    betweenness = nx.betweenness_centrality(graph, weight=None)
    closeness = nx.closeness_centrality(graph, wf_improved=True)
    components_before = nx.number_connected_components(graph)
    articulations = set(nx.articulation_points(graph))
    rows = []
    for node, attributes in graph.nodes(data=True):
        row = {"node_id": node, **attributes}
        row.update(
            degree=graph.degree(node),
            strength=graph.degree(node, weight="weight"),
            betweenness=betweenness[node],
            closeness=closeness[node],
            articulation=node in articulations,
            component_increase=0,
        )
        if node in articulations:
            reduced = graph.copy()
            reduced.remove_node(node)
            row["component_increase"] = (
                nx.number_connected_components(reduced) - components_before
            )
        rows.append(row)
    nodes = pd.DataFrame(rows)
    diversity = comments.groupby("author_channel_id").agg(
        distinct_channels=("channel_id", "nunique"),
        distinct_videos=("video_id", "nunique"),
        comment_count=("comment_id", "size"),
    )
    authors = nodes[nodes.tipo == "autor"].merge(
        diversity, left_on="node_id", right_index=True, validate="one_to_one"
    )
    authors["recurrent"] = authors.comment_count > 1
    authors["extra_comments"] = authors.comment_count - authors.distinct_videos
    videos = nodes[nodes.tipo == "video"].copy()
    projection = nx.algorithms.bipartite.weighted_projected_graph(
        graph, videos.node_id.tolist()
    )
    videos["shared_audience_videos"] = videos.node_id.map(dict(projection.degree()))
    videos["shared_audience_strength"] = videos.node_id.map(
        dict(projection.degree(weight="weight"))
    )
    videos["projection_articulation"] = videos.node_id.isin(nx.articulation_points(projection))
    order = ["betweenness", "degree", "node_id"]
    return (
        nodes,
        authors.sort_values(order, ascending=[False, False, True]),
        videos.sort_values(order, ascending=[False, False, True]),
    )


def classify_sentiment(comments, batch_size=16):
    """Pinned Spanish model, official preprocessing, deterministic CPU inference."""
    import torch
    from pysentimiento.preprocessing import preprocess_tweet
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_safetensors=True
    ).eval().to("cpu")
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    assert set(labels) == set(LABELS)
    result = comments.copy()
    result["sentiment_input"] = result.texto_original.fillna("").map(
        lambda text: preprocess_tweet(text, lang="es")
    )
    result["sentiment_status"] = np.where(
        result.sentiment_input.str.strip().eq(""), "empty_input", "classified"
    )
    result["sentiment"] = "UNCLASSIFIED"
    for label in LABELS:
        result[f"p_{label}"] = np.nan
    result["token_count"] = 0
    valid = result.index[result.sentiment_status.eq("classified")].tolist()
    for start in range(0, len(valid), batch_size):
        indices = valid[start:start + batch_size]
        texts = result.loc[indices, "sentiment_input"].tolist()
        counts = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.inference_mode():
            probabilities = model(**inputs).logits.softmax(dim=-1).numpy()
        result.loc[indices, "token_count"] = counts
        result.loc[indices, "sentiment"] = [labels[i] for i in probabilities.argmax(axis=1)]
        for position, label in enumerate(labels):
            result.loc[indices, f"p_{label}"] = probabilities[:, position]
    result["truncated"] = result.token_count > 128
    result["confidence"] = result[[f"p_{label}" for label in LABELS]].max(axis=1)
    result["low_confidence"] = result.confidence < 0.60
    result["sentiment_score"] = result.p_POS - result.p_NEG
    result["model_id"] = MODEL_ID
    result["model_revision"] = MODEL_REVISION
    return result


def sentiment_summary(comments, group, minimum_n=10):
    """Keep small groups visible; distinguish all rows from classified rows."""
    rows = []
    for key, subset in comments.groupby(group, dropna=False, sort=True):
        valid = subset[subset.sentiment_status.eq("classified")]
        counts = valid.sentiment.value_counts()
        row = {
            group: key, "n_total": len(subset), "n_classified": len(valid),
            "small_sample": len(valid) < minimum_n,
            "mean_score": valid.sentiment_score.mean(),
            "low_confidence": int(valid.low_confidence.sum()),
            "truncated": int(valid.truncated.sum()),
        }
        for label in LABELS:
            row[label] = int(counts.get(label, 0))
            row[f"pct_{label}"] = counts.get(label, 0) / len(valid) if len(valid) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_total", ascending=False)


def community_profiles(graph, comments):
    """Profile internal comments only; count inbound/outbound boundary activity."""
    result = comments.copy()
    assignment = nx.get_node_attributes(graph, "comunidad")
    result["author_community"] = result.author_channel_id.map(assignment)
    result["video_community"] = result.video_id.map(assignment)
    result["internal"] = result.author_community.eq(result.video_community)
    profiles = []
    for community in sorted(set(assignment.values())):
        nodes = [node for node, value in assignment.items() if value == community]
        internal = result[result.internal & result.video_community.eq(community)]
        videos = [node for node in nodes if graph.nodes[node]["tipo"] == "video"]
        authors = [node for node in nodes if graph.nodes[node]["tipo"] == "autor"]
        words = Counter(token for tokens in internal.tokens_limpios for token in tokens if len(token) > 2)
        profiles.append({
            "community": community, "nodes": len(nodes), "assigned_authors": len(authors),
            "assigned_videos": len(videos), "observed_internal_authors": internal.author_channel_id.nunique(),
            "internal_comments": len(internal),
            "incoming_comments": int((~result.internal & result.video_community.eq(community)).sum()),
            "outgoing_comments": int((~result.internal & result.author_community.eq(community)).sum()),
            "videos": " | ".join(graph.nodes[node]["titulo"] for node in sorted(videos)),
            "channels": " | ".join(sorted(internal.channel_name.unique())),
            "top_words": ", ".join(f"{word}:{count}" for word, count in words.most_common(8)),
        })
    profiles = pd.DataFrame(profiles).sort_values(["nodes", "community"], ascending=[False, True])
    summary = sentiment_summary(result[result.internal], "video_community")
    profiles = profiles.merge(summary, left_on="community", right_on="video_community", how="left", validate="one_to_one")
    return result, profiles


def validate_analysis(graph, author_projection, video_projection, comments, original):
    assert nx.is_bipartite(graph) and not graph.is_directed()
    assert comments.comment_id.is_unique
    assert comments.texto_original.equals(original)
    assert set(graph) == set(comments.author_channel_id) | set(comments.video_id)
    expected = comments.groupby(["author_channel_id", "video_id"]).size()
    assert graph.number_of_edges() == len(expected)
    for (author, video), weight in expected.items():
        assert graph[author][video]["weight"] == weight
        assert graph.nodes[author]["tipo"] == "autor" and graph.nodes[video]["tipo"] == "video"
    assert sum(nx.get_edge_attributes(graph, "weight").values()) == len(comments)
    for projection in (author_projection, video_projection):
        nodes = list(projection)
        for i, left in enumerate(nodes):
            for right in nodes[i + 1:]:
                shared = len(set(graph[left]) & set(graph[right]))
                assert projection.has_edge(left, right) == bool(shared)
                if shared:
                    assert projection[left][right]["weight"] == shared
    valid = comments.sentiment_status.eq("classified")
    assert np.allclose(comments.loc[valid, [f"p_{label}" for label in LABELS]].sum(axis=1), 1, atol=1e-6)
    assert comments.loc[valid, "sentiment"].isin(LABELS).all()
    assert comments.loc[valid, "confidence"].between(0, 1).all()
    assert comments[["author_community", "video_community"]].notna().all().all()
    return {"comments": len(comments), "classified": int(valid.sum()), "nodes": len(graph),
            "edges": graph.number_of_edges(), "projection_pairs_verified": sum(len(p) * (len(p) - 1) // 2 for p in (author_projection, video_projection))}


def export_results(graph, projections, tables, validation, output="results"):
    target = Path(output)
    target.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(target / f"{name}.csv", index=False, encoding="utf-8-sig")
    for name, projection in projections.items():
        nx.to_pandas_edgelist(projection).to_csv(target / f"{name}_edges.csv", index=False)
    edges = nx.to_pandas_edgelist(graph)
    for side in ("source", "target"):
        edges[f"{side}_type"] = edges[side].map(nx.get_node_attributes(graph, "tipo"))
    edges.to_csv(target / "edges.csv", index=False)
    manifest = {
        "validation": validation, "model": MODEL_ID, "revision": MODEL_REVISION,
        "max_tokens": 128, "seed": 42, "community_scope": "internal author-video comments",
        "inputs_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ("youtube_comments.csv", "youtube_videos.csv")},
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
