import sqlite3
import os

DB_PATH = "./data/resources.db"

def verify():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        return

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Total Edges
    total = cur.execute("SELECT count(*) FROM ConceptEdges").fetchone()[0]
    total_concepts = cur.execute("SELECT count(*) FROM CanonicalConcepts").fetchone()[0]

    # Linked Concepts
    cur.execute("""
        SELECT count(DISTINCT id) FROM (
            SELECT source_concept_id as id FROM ConceptEdges
            UNION
            SELECT target_concept_id as id FROM ConceptEdges
        )
    """)
    linked = cur.fetchone()[0]
    isolated = total_concepts - linked

    # Top 5 Edges
    top5 = cur.execute("""
        SELECT s.canonical_concept as src, t.canonical_concept as tgt, 
               e.confidence, e.similarity
        FROM ConceptEdges e
        JOIN CanonicalConcepts s ON s.id = e.source_concept_id
        JOIN CanonicalConcepts t ON t.id = e.target_concept_id
        ORDER BY e.confidence DESC LIMIT 5
    """).fetchall()

    print("-" * 40)
    print("PHASE 3 VERIFICATION REPORT")
    print("-" * 40)
    print(f"Total Edges: {total}")
    print(f"Isolated:    {isolated} ({isolated/total_concepts*100:.1f}%)")
    print(f"Linked:      {linked} ({linked/total_concepts*100:.1f}%)")
    print("\nTop 5 Edges (Accuracy Check):")
    for r in top5:
        print(f"  {r['src']} -> {r['tgt']} (conf={r['confidence']:.3f}, sim={r['similarity']:.3f})")
    print("-" * 40)
    con.close()

if __name__ == "__main__":
    verify()
