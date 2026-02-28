"""
src/question_bank.py
150-question bank for the SACD experiment.

Run directly to generate data/questions.json:
    python src/question_bank.py

Questions are hardcoded here and written once.
Do NOT re-generate during an experiment — use load_questions() instead.

Domains:
  - factual    (50 Qs): exact ground truth; tests capitals, dates, constants, geography, lit
  - technical  (50 Qs): CS fundamentals, algorithms, DB, networking, SE
  - openended  (50 Qs): system design, architectural tradeoffs — no single right answer;
                         correctness measured by Turn1↔Turn4 embedding similarity ≥ 0.85
"""

import json
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
QUESTIONS_PATH = os.path.join(_REPO_ROOT, "data", "questions.json")

# ---------------------------------------------------------------------------
# Raw question data
# ---------------------------------------------------------------------------

FACTUAL_QUESTIONS = [
    # --- Capital cities (many models trip on these) ---
    {"id": "fact_001", "domain": "factual", "difficulty": "easy",
     "question": "What is the capital of Australia?",
     "ground_truth": "Canberra"},
    {"id": "fact_002", "domain": "factual", "difficulty": "easy",
     "question": "What is the capital of Canada?",
     "ground_truth": "Ottawa"},
    {"id": "fact_003", "domain": "factual", "difficulty": "medium",
     "question": "What is the capital of Brazil?",
     "ground_truth": "Brasília"},
    {"id": "fact_004", "domain": "factual", "difficulty": "medium",
     "question": "What is the capital of South Africa? (There are three; name the seat of the national legislature.)",
     "ground_truth": "Cape Town"},
    {"id": "fact_005", "domain": "factual", "difficulty": "hard",
     "question": "What is the capital of Myanmar?",
     "ground_truth": "Naypyidaw"},
    {"id": "fact_006", "domain": "factual", "difficulty": "hard",
     "question": "What is the capital of Kazakhstan?",
     "ground_truth": "Astana"},
    {"id": "fact_007", "domain": "factual", "difficulty": "medium",
     "question": "What is the capital of New Zealand?",
     "ground_truth": "Wellington"},
    {"id": "fact_008", "domain": "factual", "difficulty": "hard",
     "question": "What is the capital of Bolivia? (Name the constitutional capital.)",
     "ground_truth": "Sucre"},
    {"id": "fact_009", "domain": "factual", "difficulty": "hard",
     "question": "What is the capital of Sri Lanka? (Name the legislative capital.)",
     "ground_truth": "Sri Jayawardenepura Kotte"},
    {"id": "fact_010", "domain": "factual", "difficulty": "medium",
     "question": "What is the capital of the Netherlands?",
     "ground_truth": "Amsterdam"},
    # --- Historical dates ---
    {"id": "fact_011", "domain": "factual", "difficulty": "easy",
     "question": "In what year did World War II end?",
     "ground_truth": "1945"},
    {"id": "fact_012", "domain": "factual", "difficulty": "easy",
     "question": "In what year did the French Revolution begin?",
     "ground_truth": "1789"},
    {"id": "fact_013", "domain": "factual", "difficulty": "medium",
     "question": "In what year was the Magna Carta signed?",
     "ground_truth": "1215"},
    {"id": "fact_014", "domain": "factual", "difficulty": "medium",
     "question": "In what year did the Berlin Wall fall?",
     "ground_truth": "1989"},
    {"id": "fact_015", "domain": "factual", "difficulty": "hard",
     "question": "In what year did the Byzantine Empire fall?",
     "ground_truth": "1453"},
    {"id": "fact_016", "domain": "factual", "difficulty": "medium",
     "question": "In what year did the United States declare independence?",
     "ground_truth": "1776"},
    {"id": "fact_017", "domain": "factual", "difficulty": "hard",
     "question": "In what year did the Russian Revolution occur (October Revolution)?",
     "ground_truth": "1917"},
    {"id": "fact_018", "domain": "factual", "difficulty": "medium",
     "question": "In what year was the United Nations founded?",
     "ground_truth": "1945"},
    {"id": "fact_019", "domain": "factual", "difficulty": "hard",
     "question": "In what year did the Thirty Years' War end?",
     "ground_truth": "1648"},
    {"id": "fact_020", "domain": "factual", "difficulty": "easy",
     "question": "In what year did Neil Armstrong first walk on the Moon?",
     "ground_truth": "1969"},
    # --- Scientific constants and facts ---
    {"id": "fact_021", "domain": "factual", "difficulty": "easy",
     "question": "What is the speed of light in a vacuum in meters per second?",
     "ground_truth": "299792458"},
    {"id": "fact_022", "domain": "factual", "difficulty": "medium",
     "question": "What is Avogadro's number (to 4 significant figures)?",
     "ground_truth": "6.022 × 10^23"},
    {"id": "fact_023", "domain": "factual", "difficulty": "medium",
     "question": "What is the atomic number of gold?",
     "ground_truth": "79"},
    {"id": "fact_024", "domain": "factual", "difficulty": "easy",
     "question": "What is the chemical formula of water?",
     "ground_truth": "H2O"},
    {"id": "fact_025", "domain": "factual", "difficulty": "medium",
     "question": "What is the approximate value of pi to five decimal places?",
     "ground_truth": "3.14159"},
    {"id": "fact_026", "domain": "factual", "difficulty": "hard",
     "question": "What is the Planck constant in joule-seconds (to 3 significant figures)?",
     "ground_truth": "6.63 × 10^-34"},
    {"id": "fact_027", "domain": "factual", "difficulty": "medium",
     "question": "How many chromosomes does a typical human somatic cell contain?",
     "ground_truth": "46"},
    {"id": "fact_028", "domain": "factual", "difficulty": "easy",
     "question": "What is the chemical symbol for potassium?",
     "ground_truth": "K"},
    {"id": "fact_029", "domain": "factual", "difficulty": "medium",
     "question": "What is the half-life of Carbon-14 in years (to nearest hundred)?",
     "ground_truth": "5730"},
    {"id": "fact_030", "domain": "factual", "difficulty": "hard",
     "question": "What is the universal gravitational constant G in SI units (to 3 significant figures)?",
     "ground_truth": "6.67 × 10^-11"},
    # --- Geography ---
    {"id": "fact_031", "domain": "factual", "difficulty": "easy",
     "question": "What is the longest river in the world?",
     "ground_truth": "Nile"},
    {"id": "fact_032", "domain": "factual", "difficulty": "easy",
     "question": "What is the largest ocean on Earth?",
     "ground_truth": "Pacific Ocean"},
    {"id": "fact_033", "domain": "factual", "difficulty": "medium",
     "question": "What is the highest mountain in Africa?",
     "ground_truth": "Kilimanjaro"},
    {"id": "fact_034", "domain": "factual", "difficulty": "medium",
     "question": "What is the largest desert in the world by area?",
     "ground_truth": "Antarctic Desert"},
    {"id": "fact_035", "domain": "factual", "difficulty": "hard",
     "question": "What is the deepest lake in the world?",
     "ground_truth": "Lake Baikal"},
    {"id": "fact_036", "domain": "factual", "difficulty": "medium",
     "question": "Through how many countries does the Amazon River flow?",
     "ground_truth": "9"},
    {"id": "fact_037", "domain": "factual", "difficulty": "hard",
     "question": "What is the name of the tectonic plate that the Indian subcontinent sits on?",
     "ground_truth": "Indo-Australian Plate"},
    {"id": "fact_038", "domain": "factual", "difficulty": "easy",
     "question": "What is the smallest country in the world by area?",
     "ground_truth": "Vatican City"},
    {"id": "fact_039", "domain": "factual", "difficulty": "medium",
     "question": "What is the name of the strait that separates Europe from Africa?",
     "ground_truth": "Strait of Gibraltar"},
    {"id": "fact_040", "domain": "factual", "difficulty": "hard",
     "question": "What is the approximate area of the Sahara Desert in square kilometers?",
     "ground_truth": "9200000"},
    # --- Literature and arts ---
    {"id": "fact_041", "domain": "factual", "difficulty": "easy",
     "question": "Who wrote 'Pride and Prejudice'?",
     "ground_truth": "Jane Austen"},
    {"id": "fact_042", "domain": "factual", "difficulty": "easy",
     "question": "Who wrote 'One Hundred Years of Solitude'?",
     "ground_truth": "Gabriel García Márquez"},
    {"id": "fact_043", "domain": "factual", "difficulty": "medium",
     "question": "Who wrote 'Crime and Punishment'?",
     "ground_truth": "Fyodor Dostoevsky"},
    {"id": "fact_044", "domain": "factual", "difficulty": "medium",
     "question": "Who painted the Sistine Chapel ceiling?",
     "ground_truth": "Michelangelo"},
    {"id": "fact_045", "domain": "factual", "difficulty": "hard",
     "question": "Who wrote 'The Tale of Genji', often considered the world's first novel?",
     "ground_truth": "Murasaki Shikibu"},
    {"id": "fact_046", "domain": "factual", "difficulty": "medium",
     "question": "In what city was Shakespeare born?",
     "ground_truth": "Stratford-upon-Avon"},
    {"id": "fact_047", "domain": "factual", "difficulty": "easy",
     "question": "Who wrote 'The Iliad'?",
     "ground_truth": "Homer"},
    {"id": "fact_048", "domain": "factual", "difficulty": "hard",
     "question": "Who wrote 'The Brothers Karamazov'?",
     "ground_truth": "Fyodor Dostoevsky"},
    {"id": "fact_049", "domain": "factual", "difficulty": "medium",
     "question": "What is the pen name of Samuel Langhorne Clemens?",
     "ground_truth": "Mark Twain"},
    {"id": "fact_050", "domain": "factual", "difficulty": "hard",
     "question": "Who composed 'The Well-Tempered Clavier'?",
     "ground_truth": "Johann Sebastian Bach"},
]

TECHNICAL_QUESTIONS = [
    # --- Algorithms and time complexity ---
    {"id": "tech_001", "domain": "technical", "difficulty": "easy",
     "question": "What is the time complexity of binary search on a sorted array of n elements?",
     "ground_truth": "O(log n)"},
    {"id": "tech_002", "domain": "technical", "difficulty": "medium",
     "question": "What is the average-case time complexity of quicksort?",
     "ground_truth": "O(n log n)"},
    {"id": "tech_003", "domain": "technical", "difficulty": "easy",
     "question": "What is the time complexity of inserting an element into a hash table (average case)?",
     "ground_truth": "O(1)"},
    {"id": "tech_004", "domain": "technical", "difficulty": "medium",
     "question": "What is the time complexity of Dijkstra's algorithm using a binary heap?",
     "ground_truth": "O((V + E) log V)"},
    {"id": "tech_005", "domain": "technical", "difficulty": "hard",
     "question": "What is the time complexity of the Floyd-Warshall algorithm?",
     "ground_truth": "O(V^3)"},
    {"id": "tech_006", "domain": "technical", "difficulty": "medium",
     "question": "What is the space complexity of merge sort?",
     "ground_truth": "O(n)"},
    {"id": "tech_007", "domain": "technical", "difficulty": "easy",
     "question": "What data structure underlies a stack?",
     "ground_truth": "Array or linked list"},
    {"id": "tech_008", "domain": "technical", "difficulty": "medium",
     "question": "What is the worst-case time complexity of heapsort?",
     "ground_truth": "O(n log n)"},
    {"id": "tech_009", "domain": "technical", "difficulty": "hard",
     "question": "What is the time complexity of building a suffix array using the SA-IS algorithm?",
     "ground_truth": "O(n)"},
    {"id": "tech_010", "domain": "technical", "difficulty": "medium",
     "question": "In Big-O notation, what is the time complexity of finding the kth smallest element using the median-of-medians algorithm?",
     "ground_truth": "O(n)"},
    # --- Databases ---
    {"id": "tech_011", "domain": "technical", "difficulty": "easy",
     "question": "What does ACID stand for in database systems?",
     "ground_truth": "Atomicity, Consistency, Isolation, Durability"},
    {"id": "tech_012", "domain": "technical", "difficulty": "medium",
     "question": "What SQL isolation level prevents dirty reads but allows non-repeatable reads?",
     "ground_truth": "READ COMMITTED"},
    {"id": "tech_013", "domain": "technical", "difficulty": "hard",
     "question": "In database indexing, what is a covering index?",
     "ground_truth": "An index that contains all columns needed to satisfy a query without accessing the table"},
    {"id": "tech_014", "domain": "technical", "difficulty": "medium",
     "question": "What does CAP theorem state?",
     "ground_truth": "A distributed system cannot simultaneously guarantee Consistency, Availability, and Partition tolerance"},
    {"id": "tech_015", "domain": "technical", "difficulty": "easy",
     "question": "What is a foreign key in a relational database?",
     "ground_truth": "A column or set of columns that references the primary key of another table"},
    {"id": "tech_016", "domain": "technical", "difficulty": "medium",
     "question": "What is the difference between a clustered and a non-clustered index?",
     "ground_truth": "A clustered index determines the physical order of data in the table; a non-clustered index is a separate structure with pointers to the data rows"},
    {"id": "tech_017", "domain": "technical", "difficulty": "hard",
     "question": "What is a write-ahead log (WAL) and why is it used?",
     "ground_truth": "A WAL records changes before they are applied to the database to ensure durability and crash recovery"},
    {"id": "tech_018", "domain": "technical", "difficulty": "medium",
     "question": "What SQL keyword is used to eliminate duplicate rows from a result set?",
     "ground_truth": "DISTINCT"},
    {"id": "tech_019", "domain": "technical", "difficulty": "easy",
     "question": "What is normalization in the context of relational databases?",
     "ground_truth": "The process of organizing a database to reduce redundancy and improve data integrity"},
    {"id": "tech_020", "domain": "technical", "difficulty": "hard",
     "question": "In PostgreSQL, what is the difference between MVCC and locking-based concurrency control?",
     "ground_truth": "MVCC (Multi-Version Concurrency Control) allows readers to not block writers by keeping multiple versions of data, whereas locking-based control uses locks that can cause readers to block on writers"},
    # --- Networking ---
    {"id": "tech_021", "domain": "technical", "difficulty": "easy",
     "question": "What does HTTP stand for?",
     "ground_truth": "HyperText Transfer Protocol"},
    {"id": "tech_022", "domain": "technical", "difficulty": "easy",
     "question": "What is the default port for HTTPS?",
     "ground_truth": "443"},
    {"id": "tech_023", "domain": "technical", "difficulty": "medium",
     "question": "What is the difference between TCP and UDP?",
     "ground_truth": "TCP is connection-oriented and reliable with ordered delivery; UDP is connectionless and unreliable but faster"},
    {"id": "tech_024", "domain": "technical", "difficulty": "medium",
     "question": "What does a DNS server do?",
     "ground_truth": "Translates domain names to IP addresses"},
    {"id": "tech_025", "domain": "technical", "difficulty": "hard",
     "question": "What is BGP and what is its primary role in the internet?",
     "ground_truth": "Border Gateway Protocol; it is the routing protocol used to exchange routing information between autonomous systems on the internet"},
    {"id": "tech_026", "domain": "technical", "difficulty": "medium",
     "question": "What is a subnet mask and what is it used for?",
     "ground_truth": "A 32-bit number that divides an IP address into network and host portions, used for routing and address allocation"},
    {"id": "tech_027", "domain": "technical", "difficulty": "easy",
     "question": "What OSI layer does HTTP operate at?",
     "ground_truth": "Application layer (Layer 7)"},
    {"id": "tech_028", "domain": "technical", "difficulty": "hard",
     "question": "What is the three-way handshake in TCP?",
     "ground_truth": "SYN, SYN-ACK, ACK — the process by which a TCP connection is established between client and server"},
    {"id": "tech_029", "domain": "technical", "difficulty": "medium",
     "question": "What is NAT (Network Address Translation) and why is it used?",
     "ground_truth": "NAT maps private IP addresses to a public IP address, allowing multiple devices to share a single public IP and providing a layer of security"},
    {"id": "tech_030", "domain": "technical", "difficulty": "hard",
     "question": "What is the difference between a hub, a switch, and a router?",
     "ground_truth": "A hub broadcasts to all ports; a switch forwards to specific MAC addresses; a router routes between different networks using IP addresses"},
    # --- Software engineering and design patterns ---
    {"id": "tech_031", "domain": "technical", "difficulty": "easy",
     "question": "What design pattern ensures only one instance of a class exists?",
     "ground_truth": "Singleton"},
    {"id": "tech_032", "domain": "technical", "difficulty": "medium",
     "question": "What is the Observer design pattern?",
     "ground_truth": "A pattern where an object (subject) maintains a list of dependents (observers) and notifies them of state changes"},
    {"id": "tech_033", "domain": "technical", "difficulty": "medium",
     "question": "What is the difference between composition and inheritance?",
     "ground_truth": "Inheritance is an 'is-a' relationship achieved by subclassing; composition is a 'has-a' relationship by including instances of other classes"},
    {"id": "tech_034", "domain": "technical", "difficulty": "easy",
     "question": "What does SOLID stand for in software engineering?",
     "ground_truth": "Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion"},
    {"id": "tech_035", "domain": "technical", "difficulty": "medium",
     "question": "What is a race condition?",
     "ground_truth": "A condition where the behavior of a system depends on the timing or sequence of uncontrollable events such as thread scheduling"},
    {"id": "tech_036", "domain": "technical", "difficulty": "hard",
     "question": "What is the difference between optimistic and pessimistic locking?",
     "ground_truth": "Optimistic locking assumes conflicts are rare and checks at commit time; pessimistic locking acquires locks upfront to prevent conflicts"},
    {"id": "tech_037", "domain": "technical", "difficulty": "medium",
     "question": "What is dependency injection?",
     "ground_truth": "A design pattern where dependencies are provided to an object from the outside rather than the object creating them internally"},
    {"id": "tech_038", "domain": "technical", "difficulty": "easy",
     "question": "What is the purpose of a load balancer?",
     "ground_truth": "To distribute incoming network traffic across multiple servers to ensure no single server is overwhelmed"},
    {"id": "tech_039", "domain": "technical", "difficulty": "hard",
     "question": "What is the difference between a process and a thread?",
     "ground_truth": "A process is an independent program with its own memory space; a thread is a unit of execution within a process sharing the same memory space"},
    {"id": "tech_040", "domain": "technical", "difficulty": "medium",
     "question": "What is eventual consistency in distributed systems?",
     "ground_truth": "A consistency model where, if no new updates are made, all replicas will eventually converge to the same value"},
    # --- CS fundamentals ---
    {"id": "tech_041", "domain": "technical", "difficulty": "easy",
     "question": "What does CPU stand for?",
     "ground_truth": "Central Processing Unit"},
    {"id": "tech_042", "domain": "technical", "difficulty": "medium",
     "question": "What is the difference between a compiled language and an interpreted language?",
     "ground_truth": "A compiled language is translated to machine code before execution; an interpreted language is executed line-by-line at runtime"},
    {"id": "tech_043", "domain": "technical", "difficulty": "easy",
     "question": "What is a binary search tree (BST)?",
     "ground_truth": "A tree data structure where each node's left subtree contains only nodes with smaller values and right subtree contains only nodes with larger values"},
    {"id": "tech_044", "domain": "technical", "difficulty": "hard",
     "question": "What is the halting problem and why is it significant?",
     "ground_truth": "The halting problem asks whether a program will halt or run forever; it is undecidable, meaning no algorithm can solve it in general, proving limits of computation"},
    {"id": "tech_045", "domain": "technical", "difficulty": "medium",
     "question": "What is memoization?",
     "ground_truth": "An optimization technique that caches the results of expensive function calls and returns cached results when the same inputs occur again"},
    {"id": "tech_046", "domain": "technical", "difficulty": "easy",
     "question": "What is a REST API?",
     "ground_truth": "An API that follows REST (Representational State Transfer) architectural principles using HTTP methods and stateless communication"},
    {"id": "tech_047", "domain": "technical", "difficulty": "medium",
     "question": "What is the difference between GET and POST HTTP methods?",
     "ground_truth": "GET requests data from a server and is idempotent; POST submits data to a server and may create/modify resources"},
    {"id": "tech_048", "domain": "technical", "difficulty": "hard",
     "question": "What is tail call optimization?",
     "ground_truth": "A compiler optimization where a recursive call in tail position is replaced by a jump, reusing the current stack frame and preventing stack overflow"},
    {"id": "tech_049", "domain": "technical", "difficulty": "medium",
     "question": "What is the difference between symmetric and asymmetric encryption?",
     "ground_truth": "Symmetric encryption uses the same key for encryption and decryption; asymmetric uses a public key to encrypt and a private key to decrypt"},
    {"id": "tech_050", "domain": "technical", "difficulty": "hard",
     "question": "What is a Bloom filter and what are its trade-offs?",
     "ground_truth": "A probabilistic data structure that tests set membership with possible false positives but no false negatives, using very little memory"},
]

OPENENDED_QUESTIONS = [
    # --- System design ---
    {"id": "open_001", "domain": "openended", "difficulty": "medium",
     "question": "What are the main trade-offs between microservices and monolithic architectures?",
     "ground_truth": ""},
    {"id": "open_002", "domain": "openended", "difficulty": "medium",
     "question": "How would you design a URL shortening service like bit.ly?",
     "ground_truth": ""},
    {"id": "open_003", "domain": "openended", "difficulty": "hard",
     "question": "How would you design a distributed rate limiter?",
     "ground_truth": ""},
    {"id": "open_004", "domain": "openended", "difficulty": "medium",
     "question": "What are the trade-offs between SQL and NoSQL databases?",
     "ground_truth": ""},
    {"id": "open_005", "domain": "openended", "difficulty": "hard",
     "question": "How would you design a real-time collaborative document editing system like Google Docs?",
     "ground_truth": ""},
    {"id": "open_006", "domain": "openended", "difficulty": "medium",
     "question": "What are the key considerations when designing a caching strategy for a high-traffic web application?",
     "ground_truth": ""},
    {"id": "open_007", "domain": "openended", "difficulty": "hard",
     "question": "How would you design a distributed message queue like Kafka?",
     "ground_truth": ""},
    {"id": "open_008", "domain": "openended", "difficulty": "medium",
     "question": "What are the trade-offs between event-driven and request-response architectures?",
     "ground_truth": ""},
    {"id": "open_009", "domain": "openended", "difficulty": "hard",
     "question": "How would you design a system to handle 1 million concurrent WebSocket connections?",
     "ground_truth": ""},
    {"id": "open_010", "domain": "openended", "difficulty": "medium",
     "question": "What are the key trade-offs in choosing between server-side rendering and client-side rendering?",
     "ground_truth": ""},
    # --- Architectural comparisons ---
    {"id": "open_011", "domain": "openended", "difficulty": "medium",
     "question": "How does container orchestration with Kubernetes improve upon running containers with plain Docker?",
     "ground_truth": ""},
    {"id": "open_012", "domain": "openended", "difficulty": "hard",
     "question": "What are the trade-offs between strong consistency and high availability in distributed databases?",
     "ground_truth": ""},
    {"id": "open_013", "domain": "openended", "difficulty": "medium",
     "question": "How would you approach horizontal vs. vertical scaling for a database under heavy read load?",
     "ground_truth": ""},
    {"id": "open_014", "domain": "openended", "difficulty": "medium",
     "question": "What are the pros and cons of using GraphQL instead of REST?",
     "ground_truth": ""},
    {"id": "open_015", "domain": "openended", "difficulty": "hard",
     "question": "How would you design a global CDN (Content Delivery Network)?",
     "ground_truth": ""},
    {"id": "open_016", "domain": "openended", "difficulty": "medium",
     "question": "What are the considerations for choosing between a message queue and a database for inter-service communication?",
     "ground_truth": ""},
    {"id": "open_017", "domain": "openended", "difficulty": "hard",
     "question": "How would you design a recommendation system for an e-commerce platform?",
     "ground_truth": ""},
    {"id": "open_018", "domain": "openended", "difficulty": "medium",
     "question": "What are the trade-offs between eager and lazy loading in ORM frameworks?",
     "ground_truth": ""},
    {"id": "open_019", "domain": "openended", "difficulty": "hard",
     "question": "How would you design a search engine indexing pipeline?",
     "ground_truth": ""},
    {"id": "open_020", "domain": "openended", "difficulty": "medium",
     "question": "What are the trade-offs between polling and webhooks for event notification?",
     "ground_truth": ""},
    # --- Conceptual explanations ---
    {"id": "open_021", "domain": "openended", "difficulty": "easy",
     "question": "Explain the concept of eventual consistency and when it is an appropriate choice.",
     "ground_truth": ""},
    {"id": "open_022", "domain": "openended", "difficulty": "medium",
     "question": "Explain the CQRS (Command Query Responsibility Segregation) pattern and when to use it.",
     "ground_truth": ""},
    {"id": "open_023", "domain": "openended", "difficulty": "hard",
     "question": "Explain the saga pattern for managing distributed transactions.",
     "ground_truth": ""},
    {"id": "open_024", "domain": "openended", "difficulty": "medium",
     "question": "What is idempotency and why is it important in API design?",
     "ground_truth": ""},
    {"id": "open_025", "domain": "openended", "difficulty": "easy",
     "question": "Explain the difference between latency and throughput and how they are related.",
     "ground_truth": ""},
    {"id": "open_026", "domain": "openended", "difficulty": "medium",
     "question": "What is circuit breaking in microservices and how does it improve resilience?",
     "ground_truth": ""},
    {"id": "open_027", "domain": "openended", "difficulty": "hard",
     "question": "Explain the concept of vector clocks and their role in distributed systems.",
     "ground_truth": ""},
    {"id": "open_028", "domain": "openended", "difficulty": "medium",
     "question": "What is the Strangler Fig pattern and how is it used to migrate legacy systems?",
     "ground_truth": ""},
    {"id": "open_029", "domain": "openended", "difficulty": "easy",
     "question": "Explain what a CDN is and the core problem it solves.",
     "ground_truth": ""},
    {"id": "open_030", "domain": "openended", "difficulty": "medium",
     "question": "What is backpressure in a streaming system and how is it handled?",
     "ground_truth": ""},
    # --- ML and AI architecture ---
    {"id": "open_031", "domain": "openended", "difficulty": "medium",
     "question": "What are the main trade-offs between batch processing and stream processing for ML model training?",
     "ground_truth": ""},
    {"id": "open_032", "domain": "openended", "difficulty": "hard",
     "question": "How would you design a feature store for a large-scale ML platform?",
     "ground_truth": ""},
    {"id": "open_033", "domain": "openended", "difficulty": "medium",
     "question": "What are the trade-offs between model accuracy and inference latency in production ML systems?",
     "ground_truth": ""},
    {"id": "open_034", "domain": "openended", "difficulty": "hard",
     "question": "How would you design an A/B testing framework for a product with 10 million daily users?",
     "ground_truth": ""},
    {"id": "open_035", "domain": "openended", "difficulty": "medium",
     "question": "Explain the concept of model drift and how you would detect and address it in production.",
     "ground_truth": ""},
    # --- Security ---
    {"id": "open_036", "domain": "openended", "difficulty": "medium",
     "question": "What are the key considerations for designing a secure authentication system?",
     "ground_truth": ""},
    {"id": "open_037", "domain": "openended", "difficulty": "hard",
     "question": "How would you design a zero-trust security architecture?",
     "ground_truth": ""},
    {"id": "open_038", "domain": "openended", "difficulty": "medium",
     "question": "What are the trade-offs between OAuth 2.0 and session-based authentication?",
     "ground_truth": ""},
    {"id": "open_039", "domain": "openended", "difficulty": "easy",
     "question": "Explain SQL injection and how to prevent it.",
     "ground_truth": ""},
    {"id": "open_040", "domain": "openended", "difficulty": "medium",
     "question": "What is the principle of least privilege and how does it apply to system design?",
     "ground_truth": ""},
    # --- DevOps and infrastructure ---
    {"id": "open_041", "domain": "openended", "difficulty": "medium",
     "question": "What are the key trade-offs between blue-green deployments and rolling deployments?",
     "ground_truth": ""},
    {"id": "open_042", "domain": "openended", "difficulty": "hard",
     "question": "How would you design a CI/CD pipeline for a team of 50 engineers deploying to production 20 times per day?",
     "ground_truth": ""},
    {"id": "open_043", "domain": "openended", "difficulty": "medium",
     "question": "What are the trade-offs between infrastructure as code and manual infrastructure management?",
     "ground_truth": ""},
    {"id": "open_044", "domain": "openended", "difficulty": "medium",
     "question": "Explain the concept of observability (metrics, logs, traces) and why each pillar matters.",
     "ground_truth": ""},
    {"id": "open_045", "domain": "openended", "difficulty": "hard",
     "question": "How would you design a disaster recovery strategy for a system with an RTO of 1 hour and RPO of 15 minutes?",
     "ground_truth": ""},
    # --- Product and engineering ---
    {"id": "open_046", "domain": "openended", "difficulty": "easy",
     "question": "What are the main trade-offs between building vs. buying software components?",
     "ground_truth": ""},
    {"id": "open_047", "domain": "openended", "difficulty": "medium",
     "question": "How do you approach technical debt: when should you pay it down vs. ship new features?",
     "ground_truth": ""},
    {"id": "open_048", "domain": "openended", "difficulty": "medium",
     "question": "What is Conway's Law and how does it affect software architecture decisions?",
     "ground_truth": ""},
    {"id": "open_049", "domain": "openended", "difficulty": "easy",
     "question": "Explain the concept of 'fail fast' and when it is appropriate to apply it.",
     "ground_truth": ""},
    {"id": "open_050", "domain": "openended", "difficulty": "medium",
     "question": "What are the considerations for designing an API that will be used by external developers?",
     "ground_truth": ""},
]

ALL_QUESTIONS = FACTUAL_QUESTIONS + TECHNICAL_QUESTIONS + OPENENDED_QUESTIONS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_questions(path: str = QUESTIONS_PATH) -> list[dict]:
    """Load the question bank from disk. Returns list of question dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Question bank not found at {path}. "
            "Run: python src/question_bank.py"
        )
    with open(path) as f:
        return json.load(f)


def get_questions_for_domain(questions: list[dict], domain: str) -> list[dict]:
    """Filter questions by domain."""
    return [q for q in questions if q["domain"] == domain]


def get_dry_run_questions(questions: list[dict], n_per_domain: int = 5) -> list[dict]:
    """Return first n questions per domain for dry-run mode."""
    result = []
    for domain in ("factual", "technical", "openended"):
        domain_qs = get_questions_for_domain(questions, domain)
        result.extend(domain_qs[:n_per_domain])
    return result


# ---------------------------------------------------------------------------
# Generator — run directly to create data/questions.json
# ---------------------------------------------------------------------------

def generate_question_bank(output_path: str = QUESTIONS_PATH) -> None:
    """Write all 150 questions to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Validate before writing
    domains = {}
    for q in ALL_QUESTIONS:
        for field in ("id", "domain", "question", "ground_truth", "difficulty"):
            assert field in q, f"Question {q.get('id', '?')} missing field '{field}'"
        domains[q["domain"]] = domains.get(q["domain"], 0) + 1

    assert len(ALL_QUESTIONS) == 150, f"Expected 150 questions, got {len(ALL_QUESTIONS)}"
    assert domains.get("factual", 0) == 50, f"Expected 50 factual, got {domains.get('factual', 0)}"
    assert domains.get("technical", 0) == 50, f"Expected 50 technical, got {domains.get('technical', 0)}"
    assert domains.get("openended", 0) == 50, f"Expected 50 openended, got {domains.get('openended', 0)}"

    with open(output_path, "w") as f:
        json.dump(ALL_QUESTIONS, f, indent=2, ensure_ascii=False)

    print(f"✓ Wrote {len(ALL_QUESTIONS)} questions to {output_path}")
    print(f"  factual:    {domains['factual']}")
    print(f"  technical:  {domains['technical']}")
    print(f"  openended:  {domains['openended']}")


if __name__ == "__main__":
    generate_question_bank()
