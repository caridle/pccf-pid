"""
Experiment I: Counterfactual Knowledge Conflict

Tests whether PCCF precision modulation helps LLMs follow in-context
counterfactual premises instead of defaulting to pre-training priors.

When the model is given a premise that contradicts common knowledge,
standard models tend to "hallucinate" the pre-training answer.
PCCF's reduced precision flattens attention, making the model
more reliant on in-context evidence than stored priors.
"""

COUNTERFACTUAL_TESTS = [
    # Each entry: (counterfactual_prompt, correct_answer, wrong_answer_prior)
    # correct_answer = what the context says (the "right" answer in this fictional world)
    # wrong_answer_prior = what pre-training would default to
    {
        "id": "capital_1",
        "prefix": "In an alternate universe where the capital of France was moved to Marseille in 1950,",
        "question": "What is the capital of France in this universe?",
        "answer_correct": "Marseille",
        "answer_prior": "Paris",
        "category": "geography",
    },
    {
        "id": "animal_1",
        "prefix": "Scientists have discovered that whales are actually a species of giant insect that evolved underwater.",
        "question": "According to this discovery, what category do whales belong to?",
        "answer_correct": "insect",
        "answer_prior": "mammal",
        "category": "biology",
    },
    {
        "id": "physics_1",
        "prefix": "In a simulation where gravity pushes objects upward instead of downward,",
        "question": "If you drop a ball, what happens?",
        "answer_correct": "rises",
        "answer_prior": "falls",
        "category": "physics",
    },
    {
        "id": "color_1",
        "prefix": "On planet XR-7, the sky appears green during daytime due to atmospheric methane.",
        "question": "What color is the sky on planet XR-7?",
        "answer_correct": "green",
        "answer_prior": "blue",
        "category": "planetary",
    },
    {
        "id": "language_1",
        "prefix": "In Esperanto-2, the word 'amiko' means enemy, not friend.",
        "question": "In Esperanto-2, what does 'amiko' mean?",
        "answer_correct": "enemy",
        "answer_prior": "friend",
        "category": "language",
    },
    {
        "id": "history_1",
        "prefix": "In an alternative timeline, the first person to walk on the Moon was Soviet cosmonaut Alexei Leonov in 1968.",
        "question": "Who was the first person on the Moon in this timeline?",
        "answer_correct": "Alexei Leonov",
        "answer_prior": "Neil Armstrong",
        "category": "history",
    },
    {
        "id": "math_1",
        "prefix": "In a base-5 number system where '10' represents the quantity five,",
        "question": "What does '10' represent in this system?",
        "answer_correct": "five",
        "answer_prior": "ten",
        "category": "math",
    },
    {
        "id": "food_1",
        "prefix": "In the Elven cuisine of Middle-earth, chocolate is made from mushrooms rather than cocoa beans.",
        "question": "What is Elven chocolate made from?",
        "answer_correct": "mushrooms",
        "answer_prior": "cocoa",
        "category": "food",
    },
    {
        "id": "tech_1",
        "prefix": "Apple Inc. released a new operating system called 'PearOS' that only runs on circular screens.",
        "question": "What shape screens does PearOS require?",
        "answer_correct": "circular",
        "answer_prior": "rectangular",
        "category": "technology",
    },
    {
        "id": "sport_1",
        "prefix": "In the Oceanic Games, the 100m sprint is run in a swimming pool instead of a track.",
        "question": "Where is the Oceanic 100m sprint held?",
        "answer_correct": "swimming pool",
        "answer_prior": "track",
        "category": "sports",
    },
    {
        "id": "geography_2",
        "prefix": "Due to continental drift simulations, Australia has collided with Antarctica in the year 2150.",
        "question": "What continent is Australia connected to in 2150?",
        "answer_correct": "Antarctica",
        "answer_prior": "Asia",
        "category": "geography",
    },
    {
        "id": "biology_2",
        "prefix": "Researchers found that elephants in the Congo have evolved gills and live entirely underwater.",
        "question": "What organ do Congo elephants use to breathe?",
        "answer_correct": "gills",
        "answer_prior": "lungs",
        "category": "biology",
    },
    {
        "id": "literature_1",
        "prefix": "In the unpublished sequel to Romeo and Juliet, both characters survive and open a bakery in Verona.",
        "question": "What business do Romeo and Juliet open?",
        "answer_correct": "bakery",
        "answer_prior": "they die",
        "category": "literature",
    },
    {
        "id": "music_1",
        "prefix": "Beethoven's 10th Symphony was composed using electric guitars and synthesizers.",
        "question": "What instruments did Beethoven use for his 10th Symphony?",
        "answer_correct": "electric",
        "answer_prior": "orchestra",
        "category": "music",
    },
    {
        "id": "chemistry_1",
        "prefix": "On a planet with reversed chemistry, water (H2O) is a flammable gas rather than a liquid.",
        "question": "What is the state of water on this planet?",
        "answer_correct": "gas",
        "answer_prior": "liquid",
        "category": "chemistry",
    },
    # ── Harder cases with subtler contradictions ──
    {
        "id": "hard_medicine_1",
        "prefix": "A recent WHO report reclassifies coffee as a class-A drug due to new neurological evidence.",
        "question": "According to this WHO report, how is coffee classified?",
        "answer_correct": "class-A drug",
        "answer_prior": "beverage",
        "category": "medicine",
    },
    {
        "id": "hard_econ_1",
        "prefix": "The Federal Reserve has abolished the dollar and replaced it with digital 'FedCredits'.",
        "question": "What currency has replaced the dollar?",
        "answer_correct": "digital FedCredits",
        "answer_prior": "dollar",
        "category": "economics",
    },
    {
        "id": "hard_astro_1",
        "prefix": "Astronomers reclassified Pluto as a binary planet system with its moon Charon.",
        "question": "What is Pluto now classified as?",
        "answer_correct": "binary planet",
        "answer_prior": "dwarf planet",
        "category": "astronomy",
    },
    {
        "id": "hard_law_1",
        "prefix": "The Supreme Court ruled that corporations are legally considered persons with voting rights.",
        "question": "Under this ruling, can corporations vote?",
        "answer_correct": "yes",
        "answer_prior": "no",
        "category": "law",
    },
    {
        "id": "hard_art_1",
        "prefix": "The Louvre has decided that the Mona Lisa is actually a forgery and replaced it with a photograph.",
        "question": "What is currently displayed in place of the Mona Lisa at the Louvre?",
        "answer_correct": "photograph",
        "answer_prior": "painting",
        "category": "art",
    },
    # ── Control items (no contradiction, should work the same) ──
    {
        "id": "ctrl_1",
        "prefix": "Tokyo is the capital city of Japan.",
        "question": "What is the capital of Japan?",
        "answer_correct": "Tokyo",
        "answer_prior": "Osaka",
        "category": "control",
    },
    {
        "id": "ctrl_2",
        "prefix": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "question": "At what temperature does water freeze?",
        "answer_correct": "0 degrees",
        "answer_prior": "100 degrees",
        "category": "control",
    },
    {
        "id": "ctrl_3",
        "prefix": "The Earth orbits around the Sun once every 365 days.",
        "question": "How long does it take the Earth to orbit the Sun?",
        "answer_correct": "365 days",
        "answer_prior": "24 hours",
        "category": "control",
    },
    {
        "id": "ctrl_4",
        "prefix": "Photosynthesis is the process by which plants convert sunlight into energy.",
        "question": "What process do plants use to convert sunlight into energy?",
        "answer_correct": "photosynthesis",
        "answer_prior": "respiration",
        "category": "control",
    },
    {
        "id": "ctrl_5",
        "prefix": "William Shakespeare wrote the play Hamlet.",
        "question": "Who wrote Hamlet?",
        "answer_correct": "Shakespeare",
        "answer_prior": "Dickens",
        "category": "control",
    },
]

# ── Experiment II: Few-shot Rule Shift ──

RULE_SHIFT_TASKS = [
    {
        "id": "sentiment_flip",
        "description": "Sentiment classification with mid-task label reversal",
        "before_shift": [
            ("This movie was wonderful and inspiring.", "positive"),
            ("A truly terrible experience, worst film ever.", "negative"),
            ("Absolutely loved every moment of it!", "positive"),
            ("Boring and poorly written, do not watch.", "negative"),
            ("Brilliant performance by the lead actor.", "positive"),
            ("Waste of time and money.", "negative"),
        ],
        "shift_notice": "Note: From now on, 'positive' means BAD and 'negative' means GOOD. The labels have been swapped.",
        "after_shift": [
            ("This movie was wonderful and inspiring.", "negative"),
            ("A truly terrible experience, worst film ever.", "positive"),
            ("Absolutely loved every moment of it!", "negative"),
            ("Boring and poorly written, do not watch.", "positive"),
            ("Brilliant performance by the lead actor.", "negative"),
            ("Waste of time and money.", "positive"),
        ],
        "classes": ["positive", "negative"],
    },
]

# ── Experiment III: Long Context Contradiction ──

CONTRADICTION_TESTS = [
    {
        "id": "meeting_time",
        "contexts": [
            ("The quarterly review meeting is scheduled for 3:00 PM in Conference Room A.", "early"),
            ("Please bring your laptop and the presentation slides to the quarterly review.", "filler"),
            ("Attendance is mandatory for all department heads and team leaders.", "filler"),
            ("UPDATE: The quarterly review has been moved to 5:00 PM due to a scheduling conflict.", "late"),
            ("Room remains unchanged: Conference Room A on the third floor.", "filler"),
        ],
        "question": "When is the quarterly review meeting?",
        "answer_early": "3:00 PM",
        "answer_late": "5:00 PM",
    },
    {
        "id": "ceo_change",
        "contexts": [
            ("Company press release: Sarah Chen has been appointed as the new CEO of TechCorp.", "early"),
            ("Chen brings 20 years of experience in the semiconductor industry.", "filler"),
            ("She previously served as COO of GlobalChip Manufacturing.", "filler"),
            ("CORRECTION: The board has revised its decision. Michael Torres will now serve as CEO.", "late"),
            ("The appointment is effective from next quarter.", "filler"),
        ],
        "question": "Who is the new CEO of TechCorp?",
        "answer_early": "Sarah Chen",
        "answer_late": "Michael Torres",
    },
    {
        "id": "deadline",
        "contexts": [
            ("The project deadline for the Phoenix initiative is December 15th.", "early"),
            ("All deliverables must be submitted through the project portal.", "filler"),
            ("Quality assurance review will take place in the following week.", "filler"),
            ("IMPORTANT: The deadline has been extended to December 30th per management approval.", "late"),
            ("No further extensions will be granted beyond this date.", "filler"),
        ],
        "question": "When is the project deadline?",
        "answer_early": "December 15",
        "answer_late": "December 30",
    },
    {
        "id": "flight_change",
        "contexts": [
            ("Your flight AI-302 departs from Gate 12 at 8:30 AM.", "early"),
            ("Boarding begins 45 minutes before departure.", "filler"),
            ("Please ensure you have your boarding pass and ID ready.", "filler"),
            ("GATE CHANGE: AI-302 will now depart from Gate 27.", "late"),
            ("Departure time remains unchanged at 8:30 AM.", "filler"),
        ],
        "question": "Which gate does flight AI-302 depart from?",
        "answer_early": "Gate 12",
        "answer_late": "Gate 27",
    },
    {
        "id": "budget",
        "contexts": [
            ("The marketing budget for Q3 has been set at $500,000.", "early"),
            ("Focus areas include social media campaigns and influencer partnerships.", "filler"),
            ("Regional allocations will be decided at next week's strategy meeting.", "filler"),
            ("BUDGET REVISION: Marketing budget has been reduced to $350,000 due to cost-cutting measures.", "late"),
            ("Team leads should adjust their plans accordingly.", "filler"),
        ],
        "question": "What is the Q3 marketing budget?",
        "answer_early": "$500,000",
        "answer_late": "$350,000",
    },
    {
        "id": "venue",
        "contexts": [
            ("The annual gala will be held at the Grand Plaza Hotel this year.", "early"),
            ("The event starts at 7:00 PM with a cocktail reception.", "filler"),
            ("Dress code is black tie formal.", "filler"),
            ("VENUE UPDATE: The gala has been relocated to the Riverside Convention Center.", "late"),
            ("Shuttle service will be provided from Grand Plaza to Riverside.", "filler"),
        ],
        "question": "Where will the annual gala be held?",
        "answer_early": "Grand Plaza",
        "answer_late": "Riverside",
    },
]
