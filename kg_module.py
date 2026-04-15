import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx


def normalize_entity(entity: str) -> str:
    value = entity.strip().lower()
    # Naive singularization for common plural forms.
    if value.endswith("ies") and len(value) > 3:
        value = value[:-3] + "y"
    elif value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        value = value[:-1]
    return value


class KnowledgeGraphStore:
    def __init__(
        self,
        graphml_path: Path = Path("/vault/knowledge_graph.graphml"),
        json_backup_path: Path = Path("/vault/knowledge_graph.json"),
    ) -> None:
        self.graphml_path = graphml_path
        self.json_backup_path = json_backup_path
        self.graphml_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph: nx.DiGraph = self.load()

    def load(self) -> nx.DiGraph:
        if self.graphml_path.exists():
            loaded = nx.read_graphml(self.graphml_path)
            graph = nx.DiGraph()
            graph.add_nodes_from(loaded.nodes(data=True))
            graph.add_edges_from(loaded.edges(data=True))
            return graph
        return nx.DiGraph()

    def save(self) -> None:
        nx.write_graphml(self.graph, self.graphml_path)
        backup = {
            "nodes": [{"id": n, **attrs} for n, attrs in self.graph.nodes(data=True)],
            "edges": [
                {
                    "subject": u,
                    "object": v,
                    **attrs,
                }
                for u, v, attrs in self.graph.edges(data=True)
            ],
            "metadata": {
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
        }
        with self.json_backup_path.open("w", encoding="utf-8") as f:
            json.dump(backup, f, indent=2, ensure_ascii=False)

    def add_triples(self, triples: List[Dict[str, Any]], source_file: Optional[str] = None) -> int:
        added = 0
        timestamp = datetime.now(timezone.utc).isoformat()
        source = source_file or "input_stream"

        for triple in triples:
            subject = normalize_entity(str(triple.get("subject", "")))
            predicate = str(triple.get("predicate", "")).strip().lower()
            obj = normalize_entity(str(triple.get("object", "")))
            confidence = float(triple.get("confidence", 0.0))
            if not subject or not predicate or not obj:
                continue

            if subject not in self.graph:
                self.graph.add_node(subject, label=subject, source_file=source, timestamp=timestamp)
            if obj not in self.graph:
                self.graph.add_node(obj, label=obj, source_file=source, timestamp=timestamp)

            if self.graph.has_edge(subject, obj):
                existing_predicate = str(self.graph[subject][obj].get("predicate", "")).lower()
                if existing_predicate == predicate:
                    continue

            self.graph.add_edge(
                subject,
                obj,
                predicate=predicate,
                confidence=confidence,
                source_file=source,
                timestamp=timestamp,
            )
            added += 1

        if added:
            self.save()
        return added

    def query_entity(self, entity: str) -> List[Dict[str, Any]]:
        node = normalize_entity(entity)
        if node not in self.graph:
            return []
        relationships: List[Dict[str, Any]] = []
        for neighbor in self.graph.successors(node):
            edge_data = self.graph.get_edge_data(node, neighbor) or {}
            relationships.append(
                {
                    "subject": node,
                    "predicate": edge_data.get("predicate", ""),
                    "object": neighbor,
                    "confidence": edge_data.get("confidence", 0.0),
                }
            )
        for predecessor in self.graph.predecessors(node):
            edge_data = self.graph.get_edge_data(predecessor, node) or {}
            relationships.append(
                {
                    "subject": predecessor,
                    "predicate": edge_data.get("predicate", ""),
                    "object": node,
                    "confidence": edge_data.get("confidence", 0.0),
                }
            )
        return relationships

    def find_relationship(self, subject: str, predicate: str) -> List[Dict[str, Any]]:
        subj = normalize_entity(subject)
        pred = predicate.strip().lower()
        if subj not in self.graph:
            return []
        found: List[Dict[str, Any]] = []
        for obj in self.graph.successors(subj):
            edge_data = self.graph.get_edge_data(subj, obj) or {}
            if str(edge_data.get("predicate", "")).lower() == pred:
                found.append(
                    {
                        "subject": subj,
                        "predicate": pred,
                        "object": obj,
                        "confidence": edge_data.get("confidence", 0.0),
                    }
                )
        return found
