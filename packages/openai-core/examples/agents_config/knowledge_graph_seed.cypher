// ---------------------------------------------------------------------------
// Seed data for the Neo4j knowledge-graph demo (insurance domain).
// Matches knowledge_graph_agent.yaml: labels Policy|Coverage|Customer|Claim
// and a full-text index named `entityNames`.
//
// Load it with:
//   docker exec -i neo4j-adk cypher-shell -u neo4j -p testpassword123 \
//     < knowledge_graph_seed.cypher
// ---------------------------------------------------------------------------

// --- Customers ---
MERGE (acme:Customer {id: 'C-001', name: 'Acme Corp',
       description: 'Commercial customer headquartered in Dubai'});
MERGE (jane:Customer {id: 'C-002', name: 'Jane Doe',
       description: 'Retail customer, individual policyholder'});

// --- Policies ---
MERGE (p1:Policy {id: 'P-1001', name: 'Home Shield 360',
       description: 'Comprehensive home insurance policy with fire, flood and theft cover'});
MERGE (p2:Policy {id: 'P-1002', name: 'Auto Secure',
       description: 'Motor insurance policy covering collision and third-party liability'});

// --- Coverages ---
MERGE (fire:Coverage  {id: 'COV-01', name: 'Fire Damage',  description: 'Covers damage caused by fire', limit: 500000});
MERGE (flood:Coverage {id: 'COV-02', name: 'Flood Damage', description: 'Covers damage caused by flooding', limit: 250000});
MERGE (theft:Coverage {id: 'COV-03', name: 'Theft',        description: 'Covers loss of property due to theft', limit: 100000});
MERGE (coll:Coverage  {id: 'COV-04', name: 'Collision',    description: 'Covers vehicle collision damage', limit: 150000});
MERGE (liab:Coverage  {id: 'COV-05', name: 'Liability',    description: 'Covers third-party liability claims', limit: 1000000});

// --- Claims ---
MERGE (clm:Claim {id: 'CLM-5001', name: 'Kitchen fire claim',
       description: 'Fire damage claim filed for the Home Shield 360 policy', status: 'under_review', amount: 42000});

// --- Relationships ---
MATCH (acme:Customer {id:'C-001'}), (jane:Customer {id:'C-002'}),
      (p1:Policy {id:'P-1001'}), (p2:Policy {id:'P-1002'}),
      (fire:Coverage {id:'COV-01'}), (flood:Coverage {id:'COV-02'}),
      (theft:Coverage {id:'COV-03'}), (coll:Coverage {id:'COV-04'}),
      (liab:Coverage {id:'COV-05'}), (clm:Claim {id:'CLM-5001'})
MERGE (jane)-[:HOLDS]->(p1)
MERGE (acme)-[:HOLDS]->(p2)
MERGE (p1)-[:HAS_COVERAGE]->(fire)
MERGE (p1)-[:HAS_COVERAGE]->(flood)
MERGE (p1)-[:HAS_COVERAGE]->(theft)
MERGE (p2)-[:HAS_COVERAGE]->(coll)
MERGE (p2)-[:HAS_COVERAGE]->(liab)
MERGE (clm)-[:FILED_AGAINST]->(p1)
MERGE (clm)-[:FILED_BY]->(jane)
MERGE (clm)-[:RELATES_TO]->(fire);

// --- Full-text index for entry-node lookup (name must match the config) ---
CREATE FULLTEXT INDEX entityNames IF NOT EXISTS
FOR (n:Policy|Coverage|Customer|Claim)
ON EACH [n.name, n.description];
