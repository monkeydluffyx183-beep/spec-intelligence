"""WS3 — graph storage, the query API and exports.

Store choice (justified in ``docs/KG_SCHEMA.md`` and the build report): **SQLite**
by default.  The graph is small (tens of thousands of edges at most), the queries
the agent needs are neighbourhood and path lookups, and SQLite needs no server —
which matches the assignment's "local-first, reproducible from a clean clone"
constraint.  The same class exports JSON, GraphML and Cypher, so a team that
prefers Neo4j, Memgraph or Kùzu can load the graph and keep the same helpers
(``NetworkXStore`` is provided for path queries on the same data).

Query API (contract with WS4):

    lookup(name, node_types=None)            → nodes
    neighbors(node_id, edge_types, direction, depth) → nodes + edges
    path(src, dst, max_len)                  → list of node id paths
    subgraph(seeds, depth)                   → nodes + edges for visualisation
    stats()                                  → counts by type, orphans, merges
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from specintel.common.models import Edge, GraphQuery, GraphResult, Node
from specintel.kg.resolve import EntityResolver

NODE_COLUMNS = ("node_id", "node_type", "name", "canonical_name", "doc_id", "revision", "section_number",
                "page", "aliases", "properties", "element_ids")
EDGE_COLUMNS = ("edge_id", "src_id", "dst_id", "edge_type", "doc_id", "revision", "section_number", "page",
                "element_id", "extraction_method", "confidence", "stale", "evidence", "properties")


@dataclass
class GraphStats:
    nodes: int = 0
    edges: int = 0
    nodes_by_type: dict[str, int] = field(default_factory=dict)
    edges_by_type: dict[str, int] = field(default_factory=dict)
    edges_by_method: dict[str, int] = field(default_factory=dict)
    orphan_nodes: int = 0
    stale_references: int = 0
    merges: int = 0
    duplicate_rate: float = 0.0
    cross_document_edges: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes, "edges": self.edges, "nodes_by_type": dict(sorted(self.nodes_by_type.items())),
            "edges_by_type": dict(sorted(self.edges_by_type.items())),
            "edges_by_method": dict(sorted(self.edges_by_method.items())),
            "orphan_nodes": self.orphan_nodes, "stale_references": self.stale_references,
            "merges": self.merges, "duplicate_rate": round(self.duplicate_rate, 4),
            "cross_document_edges": self.cross_document_edges,
        }


class GraphStore:
    """SQLite-backed property graph with the agent-facing query API."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        # NB: the attribute is ``db_path``, not ``path``: ``path`` is the graph
        # traversal method below and an attribute of that name would shadow it.
        self.db_path = str(path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    # -- schema ------------------------------------------------------------ #
    def _create_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY, node_type TEXT, name TEXT, canonical_name TEXT, doc_id TEXT,
            revision TEXT, section_number TEXT, page INTEGER, aliases TEXT, properties TEXT, element_ids TEXT
        );
        CREATE TABLE IF NOT EXISTS edges (
            edge_id TEXT PRIMARY KEY, src_id TEXT, dst_id TEXT, edge_type TEXT, doc_id TEXT, revision TEXT,
            section_number TEXT, page INTEGER, element_id TEXT, extraction_method TEXT, confidence REAL,
            stale INTEGER, evidence TEXT, properties TEXT
        );
        CREATE INDEX IF NOT EXISTS nodes_name ON nodes(name);
        CREATE INDEX IF NOT EXISTS nodes_canonical ON nodes(canonical_name);
        CREATE INDEX IF NOT EXISTS nodes_type ON nodes(node_type);
        CREATE INDEX IF NOT EXISTS edges_src ON edges(src_id);
        CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst_id);
        CREATE INDEX IF NOT EXISTS edges_type ON edges(edge_type);
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)
        self.conn.commit()

    def reset(self) -> None:
        """Empty the store.

        A rebuild must be idempotent: without this, a second ``build_graph`` run
        would leave the previous run's nodes and edges in place and the counts
        (and triple precision) would be measured on a mixture of two builds.
        """
        self.conn.executescript("DELETE FROM nodes; DELETE FROM edges; DELETE FROM meta;")
        self.conn.commit()

    # -- write ------------------------------------------------------------- #
    def add_nodes(self, nodes: Iterable[Node]) -> int:
        rows = [(n.node_id, n.node_type, n.name, n.canonical_name, n.doc_id, n.revision, n.section_number,
                 n.page, json.dumps(n.aliases), json.dumps(n.properties, default=str),
                 json.dumps(n.element_ids)) for n in nodes]
        self.conn.executemany(
            f"INSERT OR REPLACE INTO nodes ({','.join(NODE_COLUMNS)}) VALUES ({','.join('?' * len(NODE_COLUMNS))})",
            rows)
        self.conn.commit()
        return len(rows)

    def add_edges(self, edges: Iterable[Edge]) -> int:
        rows = [(e.edge_id, e.src_id, e.dst_id, e.edge_type, e.doc_id, e.revision, e.section_number, e.page,
                 e.element_id, e.extraction_method, e.confidence, int(bool(e.stale)), e.evidence,
                 json.dumps(e.properties, default=str)) for e in edges]
        self.conn.executemany(
            f"INSERT OR REPLACE INTO edges ({','.join(EDGE_COLUMNS)}) VALUES ({','.join('?' * len(EDGE_COLUMNS))})",
            rows)
        self.conn.commit()
        return len(rows)

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value, default=str)))
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # -- read -------------------------------------------------------------- #
    @staticmethod
    def _node(row: sqlite3.Row) -> Node:
        return Node(node_id=row["node_id"], node_type=row["node_type"], name=row["name"],
                    canonical_name=row["canonical_name"] or row["name"], doc_id=row["doc_id"] or "",
                    revision=row["revision"] or "", section_number=row["section_number"] or "",
                    page=int(row["page"] or 0), aliases=json.loads(row["aliases"] or "[]"),
                    properties=json.loads(row["properties"] or "{}"),
                    element_ids=json.loads(row["element_ids"] or "[]"))

    @staticmethod
    def _edge(row: sqlite3.Row) -> Edge:
        return Edge(edge_id=row["edge_id"], src_id=row["src_id"], dst_id=row["dst_id"],
                    edge_type=row["edge_type"], doc_id=row["doc_id"] or "", revision=row["revision"] or "",
                    section_number=row["section_number"] or "", page=int(row["page"] or 0),
                    element_id=row["element_id"] or "", extraction_method=row["extraction_method"] or "rule",
                    confidence=float(row["confidence"] or 1.0), stale=bool(row["stale"]),
                    evidence=row["evidence"] or "", properties=json.loads(row["properties"] or "{}"))

    def get_node(self, node_id: str) -> Node | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        return self._node(row) if row else None

    def lookup(self, name: str, node_types: Sequence[str] | None = None, limit: int = 10) -> list[Node]:
        """Resolve a user's wording to canonical nodes."""
        rows = self.conn.execute("SELECT * FROM nodes").fetchall()
        nodes = [self._node(r) for r in rows]
        matched = EntityResolver.lookup(nodes, name, node_types)
        return matched[:limit]

    def neighbors(self, node_id: str, edge_types: Sequence[str] | None = None, direction: str = "both",
                  depth: int = 1, limit: int = 200) -> tuple[list[Node], list[Edge]]:
        """BFS neighbourhood with per-hop edge filtering."""
        seen_nodes: dict[str, Node] = {}
        collected: list[Edge] = []
        start = self.get_node(node_id)
        if start is None:
            return [], []
        seen_nodes[node_id] = start
        frontier = [node_id]
        for _ in range(max(1, depth)):
            next_frontier: list[str] = []
            for current in frontier:
                clauses, params = [], []
                if direction in ("out", "both"):
                    clauses.append("src_id = ?")
                    params.append(current)
                if direction in ("in", "both"):
                    clauses.append("dst_id = ?")
                    params.append(current)
                sql = f"SELECT * FROM edges WHERE ({' OR '.join(clauses)})"
                if edge_types:
                    sql += f" AND edge_type IN ({','.join('?' * len(edge_types))})"
                    params.extend(edge_types)
                for row in self.conn.execute(sql, params):
                    edge = self._edge(row)
                    if edge.edge_id in {e.edge_id for e in collected}:
                        continue
                    collected.append(edge)
                    other_id = edge.dst_id if edge.src_id == current else edge.src_id
                    if other_id not in seen_nodes:
                        other = self.get_node(other_id)
                        if other is not None:
                            seen_nodes[other_id] = other
                            next_frontier.append(other_id)
            frontier = next_frontier
            if not frontier or len(collected) >= limit:
                break
        return list(seen_nodes.values()), collected[:limit]

    def path(self, src: str, dst: str, max_len: int = 4, edge_types: Sequence[str] | None = None
             ) -> list[list[str]]:
        """Shortest paths between two nodes (BFS over the directed graph)."""
        if src == dst:
            return [[src]]
        queue: deque[list[str]] = deque([[src]])
        found: list[list[str]] = []
        while queue:
            current = queue.popleft()
            if len(current) > max_len:
                continue
            last = current[-1]
            sql = "SELECT dst_id FROM edges WHERE src_id = ?"
            params: list[Any] = [last]
            if edge_types:
                sql += f" AND edge_type IN ({','.join('?' * len(edge_types))})"
                params.extend(edge_types)
            for row in self.conn.execute(sql, params):
                nxt = row["dst_id"]
                if nxt in current:
                    continue
                if nxt == dst:
                    found.append(current + [nxt])
                    if len(found) >= 3:
                        return found
                    continue
                queue.append(current + [nxt])
        return found

    def all_edges(self, edge_types: Sequence[str] | None = None, limit: int | None = None) -> list[Edge]:
        sql = "SELECT * FROM edges"
        params: list[Any] = []
        if edge_types:
            sql += f" WHERE edge_type IN ({','.join('?' * len(edge_types))})"
            params.extend(edge_types)
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [self._edge(r) for r in self.conn.execute(sql, params)]

    def query(self, request: GraphQuery) -> GraphResult:
        """Execute the typed :class:`GraphQuery` contract used by the agent."""
        result = GraphResult(query=request)
        if request.kind == "lookup":
            result.nodes = self.lookup(request.name, node_types=list(request.node_types or []) or None,
                                       limit=request.limit)
            if request.name and not result.nodes:
                result.notes.append(f"no entity matched '{request.name}'")
            return result
        if request.kind == "neighbors":
            node_id = request.node_id
            if not node_id and request.name:
                matches = self.lookup(request.name, node_types=list(request.node_types or []) or None, limit=1)
                if not matches:
                    result.notes.append(f"unknown entity '{request.name}'")
                    return result
                node_id = matches[0].node_id
            nodes, edges = self.neighbors(node_id, edge_types=list(request.edge_types or []) or None,
                                          direction=request.direction, depth=request.depth,
                                          limit=request.limit)
            result.nodes, result.edges = nodes, edges
            return result
        if request.kind == "path":
            src = _resolve_id(self, request.src)
            dst = _resolve_id(self, request.dst)
            if not src or not dst:
                result.notes.append(f"could not resolve '{request.src or request.dst}'")
                return result
            paths = self.path(src, dst, max_len=max(2, request.depth + 2),
                              edge_types=list(request.edge_types or []) or None)
            result.paths = paths
            ids = {nid for p in paths for nid in p}
            result.nodes = [n for n in (self.get_node(i) for i in ids) if n]
            result.edges = [e for e in self.all_edges() if e.src_id in ids and e.dst_id in ids]
            if not paths:
                result.notes.append(f"no path within {max(2, request.depth + 2)} hops from '{request.src}' to "
                                    f"'{request.dst}'")
            return result
        if request.kind == "subgraph":
            seeds = [request.node_id] if request.node_id else [
                n.node_id for n in self.lookup(request.name, limit=3)] if request.name else []
            nodes: dict[str, Node] = {}
            edges: list[Edge] = []
            for seed in seeds:
                sub_nodes, sub_edges = self.neighbors(seed, edge_types=list(request.edge_types or []) or None,
                                                      depth=request.depth, limit=request.limit)
                nodes.update({n.node_id: n for n in sub_nodes})
                edges.extend(sub_edges)
            result.nodes, result.edges = list(nodes.values()), edges
            return result
        if request.kind == "stats":
            result.stats = self.stats().to_dict()
            return result
        result.notes.append(f"unknown query kind '{request.kind}'")
        return result

    # -- statistics -------------------------------------------------------- #
    def stats(self, *, merges: int = 0, duplicate_rate: float = 0.0) -> GraphStats:
        stats = GraphStats()
        stats.nodes = self.conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"]
        stats.edges = self.conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"]
        for row in self.conn.execute("SELECT node_type, COUNT(*) AS c FROM nodes GROUP BY node_type"):
            stats.nodes_by_type[row["node_type"]] = row["c"]
        for row in self.conn.execute("SELECT edge_type, COUNT(*) AS c FROM edges GROUP BY edge_type"):
            stats.edges_by_type[row["edge_type"]] = row["c"]
        for row in self.conn.execute("SELECT extraction_method, COUNT(*) AS c FROM edges GROUP BY extraction_method"):
            stats.edges_by_method[row["extraction_method"]] = row["c"]
        stats.orphan_nodes = self.conn.execute(
            "SELECT COUNT(*) AS c FROM nodes WHERE node_id NOT IN (SELECT src_id FROM edges) "
            "AND node_id NOT IN (SELECT dst_id FROM edges)").fetchone()["c"]
        stats.stale_references = self.conn.execute("SELECT COUNT(*) AS c FROM edges WHERE stale = 1").fetchone()["c"]
        stats.cross_document_edges = self.conn.execute(
            "SELECT COUNT(*) AS c FROM edges e JOIN nodes s ON s.node_id = e.src_id "
            "JOIN nodes d ON d.node_id = e.dst_id WHERE d.doc_id <> '' AND s.doc_id <> '' "
            "AND d.doc_id <> s.doc_id").fetchone()["c"]
        stats.merges = merges or int(self.get_meta("merges", 0) or 0)
        stats.duplicate_rate = duplicate_rate or float(self.get_meta("duplicate_rate", 0.0) or 0.0)
        return stats

    # -- exports ----------------------------------------------------------- #
    def export_json(self, path: str | Path) -> tuple[Path, Path]:
        """Nodes and edges as JSONL (the format the reviewers can diff)."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        node_path, edge_path = p / "nodes.jsonl", p / "edges.jsonl"
        with node_path.open("w", encoding="utf-8") as fh:
            for row in self.conn.execute("SELECT * FROM nodes"):
                fh.write(self._node(row).to_jsonl())
        with edge_path.open("w", encoding="utf-8") as fh:
            for row in self.conn.execute("SELECT * FROM edges"):
                fh.write(self._edge(row).to_jsonl())
        return node_path, edge_path

    def export_graphml(self, path: str | Path) -> Path:
        """GraphML for Gephi/yEd-style visualisation."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            fh.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
            fh.write('  <key id="node_type" for="node" attr.name="node_type" attr.type="string"/>\n')
            fh.write('  <key id="doc_id" for="node" attr.name="doc_id" attr.type="string"/>\n')
            fh.write('  <key id="page" for="node" attr.name="page" attr.type="int"/>\n')
            fh.write('  <key id="edge_type" for="edge" attr.name="edge_type" attr.type="string"/>\n')
            fh.write('  <key id="confidence" for="edge" attr.name="confidence" attr.type="double"/>\n')
            fh.write('  <graph edgedefault="directed">\n')
            for row in self.conn.execute("SELECT * FROM nodes"):
                node = self._node(row)
                fh.write(f'    <node id="{_xml(node.node_id)}">\n')
                fh.write(f'      <data key="node_type">{_xml(node.node_type)}</data>\n')
                fh.write(f'      <data key="doc_id">{_xml(node.doc_id)}</data>\n')
                fh.write(f'      <data key="page">{node.page}</data>\n    </node>\n')
            for i, row in enumerate(self.conn.execute("SELECT * FROM edges")):
                edge = self._edge(row)
                fh.write(f'    <edge id="e{i}" source="{_xml(edge.src_id)}" target="{_xml(edge.dst_id)}">\n')
                fh.write(f'      <data key="edge_type">{_xml(edge.edge_type)}</data>\n')
                fh.write(f'      <data key="confidence">{edge.confidence}</data>\n    </edge>\n')
            fh.write('  </graph>\n</graphml>\n')
        return p

    def export_cypher(self, path: str | Path, *, limit: int | None = None) -> Path:
        """Cypher statements so the graph can be loaded into Neo4j/Memgraph/Kùzu."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            fh.write("// generated by specintel.kg.store — load into Neo4j with cypher-shell\n")
            for row in self.conn.execute("SELECT * FROM nodes"):
                node = self._node(row)
                props = {"name": node.name, "doc": node.doc_id, "page": node.page,
                         "section": node.section_number, "revision": node.revision}
                fh.write(f"MERGE (n:{node.node_type} {{node_id: '{_cypher(node.node_id)}'}}) "
                         f"SET n += {_cypher_props(props)};\n")
            for row in self.conn.execute("SELECT * FROM edges"):
                edge = self._edge(row)
                props = {"page": edge.page, "section": edge.section_number, "doc": edge.doc_id,
                         "method": edge.extraction_method, "confidence": edge.confidence, "stale": edge.stale}
                fh.write(f"MATCH (a {{node_id: '{_cypher(edge.src_id)}'}}), (b {{node_id: '{_cypher(edge.dst_id)}'}}) "
                         f"MERGE (a)-[r:{edge.edge_type}]->(b) SET r += {_cypher_props(props)};\n")
        return p

    def to_networkx(self) -> Any:
        """Expose the graph as a NetworkX DiGraph (optional dependency)."""
        try:
            import networkx as nx  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("networkx is not installed: pip install networkx") from exc
        graph = nx.DiGraph()
        for row in self.conn.execute("SELECT * FROM nodes"):
            node = self._node(row)
            graph.add_node(node.node_id, node_type=node.node_type, name=node.name, doc_id=node.doc_id,
                           page=node.page, section=node.section_number)
        for row in self.conn.execute("SELECT * FROM edges"):
            edge = self._edge(row)
            graph.add_edge(edge.src_id, edge.dst_id, edge_type=edge.edge_type, confidence=edge.confidence,
                           page=edge.page, doc_id=edge.doc_id, stale=edge.stale)
        return graph

    def close(self) -> None:
        self.conn.close()

    # -- helpers ----------------------------------------------------------- #
    def edge_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"])

    def node_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"])


class NetworkXStore:
    """NetworkX-backed store exposing the same query API (optional backend).

    Useful for path/centrality experiments from the graph-theory lesson, and as
    a cross-check that the SQLite path queries agree with NetworkX.
    """

    def __init__(self, graph: Any | None = None) -> None:
        import networkx as nx  # type: ignore

        self.graph = graph if graph is not None else nx.DiGraph()

    @classmethod
    def from_sqlite(cls, store: GraphStore) -> "NetworkXStore":
        return cls(store.to_networkx())

    def neighbors(self, node_id: str, edge_types: Sequence[str] | None = None, direction: str = "both",
                  depth: int = 1, limit: int = 200) -> tuple[list[Node], list[Edge]]:
        import networkx as nx  # type: ignore

        if node_id not in self.graph:
            return [], []
        visited = {node_id}
        frontier = {node_id}
        for _ in range(max(1, depth)):
            nxt: set[str] = set()
            for node in frontier:
                edges = []
                if direction in ("out", "both"):
                    edges.extend(self.graph.out_edges(node, data=True))
                if direction in ("in", "both"):
                    edges.extend(self.graph.in_edges(node, data=True))
                for src, dst, data in edges:
                    if edge_types and data.get("edge_type") not in edge_types:
                        continue
                    other = dst if src == node else src
                    nxt.add(other)
            new = nxt - visited
            if not new:
                break
            visited |= new
            frontier = new
        sub = self.graph.subgraph(visited)
        nodes = [Node(node_id=n, node_type=d.get("node_type", ""), name=d.get("name", n),
                      doc_id=d.get("doc_id", ""), page=int(d.get("page", 0) or 0),
                      section_number=d.get("section", "")) for n, d in sub.nodes(data=True)]
        edges = [Edge(edge_id=f"{s}->{t}:{d.get('edge_type')}", src_id=s, dst_id=t,
                      edge_type=d.get("edge_type", ""), doc_id=d.get("doc_id", ""),
                      page=int(d.get("page", 0) or 0), confidence=float(d.get("confidence", 1.0)),
                      stale=bool(d.get("stale"))) for s, t, d in sub.edges(data=True)]
        return nodes[:limit], edges[:limit]

    def path(self, src: str, dst: str, max_len: int = 4, edge_types: Sequence[str] | None = None
             ) -> list[list[str]]:
        import networkx as nx  # type: ignore

        if src not in self.graph or dst not in self.graph:
            return []
        try:
            paths = list(nx.all_shortest_paths(self.graph, src, dst))
        except Exception:
            return []
        return [p for p in paths if len(p) <= max_len + 1][:3]


def _resolve_id(store: GraphStore, name: str) -> str | None:
    if not name:
        return None
    node = store.get_node(name)
    if node is not None:
        return name
    matches = store.lookup(name, limit=1)
    return matches[0].node_id if matches else None


def _xml(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _cypher(text: str) -> str:
    return str(text).replace("'", "\\'")


def _cypher_props(props: dict[str, Any]) -> str:
    parts = []
    for key, value in props.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            parts.append(f"{key}: {str(value).lower()}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}: {value}")
        else:
            parts.append(f"{key}: '{_cypher(value)}'")
    return "{" + ", ".join(parts) + "}"


def visualise_subgraph(store: GraphStore, seed: str, path: str | Path, *, depth: int = 2,
                       edge_types: Sequence[str] | None = None, title: str = "") -> Path:
    """Write a Mermaid diagram of a neighbourhood (renders on GitHub)."""
    result = store.query(GraphQuery(kind="subgraph", name=seed, depth=depth,
                                    edge_types=tuple(edge_types or ()), limit=80))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"title: {title or seed}", "---", "flowchart LR"]
    ids: dict[str, str] = {}
    for node in result.nodes:
        alias = f"n{len(ids)}"
        ids[node.node_id] = alias
        label = (node.canonical_name or node.name).replace('"', "'")
        shape = {"State": f'{alias}(["{label}"])', "Timer": f'{alias}[/"{label}"/]',
                 "TestCase": f'{alias}[["{label}"]]', "Check": f'{alias}("{label}")'}.get(
            node.node_type, f'{alias}["{label}"]')
        lines.append(f"    {shape}")
    for edge in result.edges:
        if edge.src_id in ids and edge.dst_id in ids:
            lines.append(f'    {ids[edge.src_id]} -->|{edge.edge_type}| {ids[edge.dst_id]}')
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
