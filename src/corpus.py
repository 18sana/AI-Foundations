import os
from pathlib import Path
from src.config import CORPUS_DIR

DOCUMENTS = {
    "doc_01_voyager_mission.txt": (
        "Voyager 1 and Voyager 2 are twin spacecraft launched by NASA in 1977. "
        "Their primary mission was the exploration of Jupiter and Saturn. "
        "After making a string of important discoveries there, such as active volcanoes "
        "on Jupiter's moon Io and the intricacies of Saturn's rings, the mission was extended. "
        "Voyager 1 crossed the heliopause and entered interstellar space in August 2012, "
        "making it the first human-made object to do so. It continues to transmit scientific data "
        "back to Earth, though its power sources are slowly depleting."
    ),
    "doc_02_voyager_controversy.txt": (
        "While NASA officially announced that Voyager 1 crossed the heliopause in August 2012, "
        "some astrophysicists disputed the exact timing. They argued that the magnetic field data "
        "did not show the expected shift in direction. A subsequent analysis of plasma density waves "
        "published in 2013 suggested that the entry into interstellar space actually occurred in "
        "May 2013 when a solar storm caused the surrounding plasma to vibrate, providing a clear "
        "measurement of the local density."
    ),
    "doc_03_quantum_supremacy.txt": (
        "Quantum supremacy, or quantum advantage, is the demonstration that a programmable quantum computer "
        "can solve a problem that no classical computer can solve in any feasible amount of time. "
        "In 2019, Google claimed to have achieved quantum supremacy using its Sycamore processor, "
        "performing a specific calculation in 200 seconds that would take a state-of-the-art supercomputer "
        "10,000 years. However, IBM disputed this claim, showing that with advanced classical algorithms "
        "and storage techniques, the same task could be done on a classical supercomputer in 2.5 days."
    ),
    "doc_04_cap_theorem.txt": (
        "The CAP theorem, also known as Brewer's theorem, states that a distributed data store can "
        "simultaneously provide at most two of three guarantees: Consistency, Availability, and "
        "Partition Tolerance. Consistency means every read receives the most recent write or an error. "
        "Availability means every non-failing node returns a non-error response, without guarantee of "
        "it being the latest. Partition Tolerance means the system continues to operate despite arbitrary "
        "message loss or delay. In practice, networks are never 100% reliable, so systems must choose "
        "between Consistency and Availability when a partition occurs."
    ),
    "doc_05_raft_consensus.txt": (
        "Raft is a consensus algorithm designed as an alternative to Paxos. It is intended to be "
        "easier to understand than Paxos through decomposition of consensus into subproblems: "
        "leader election, log replication, and safety. Raft uses a strong leader-based model where "
        "the leader manages the replicated log. If a leader fails, a new leader is elected through "
        "a randomized election timer mechanism, ensuring cluster consistency."
    ),
    "doc_06_transformer_architecture.txt": (
        "The Transformer is a deep learning model introduced in the 2017 paper 'Attention Is All You Need'. "
        "Unlike older recurrent neural networks (RNNs), the Transformer relies entirely on self-attention "
        "mechanisms to model global dependencies between input and output tokens. This allows for "
        "significantly more parallelization during training. The core components include multi-head self-attention, "
        "positional encodings, and feed-forward neural layers, forming the foundation of modern Large Language Models."
    ),
    "doc_07_climate_change_temp.txt": (
        "Global surface temperatures have risen significantly since the pre-industrial era. "
        "According to NASA's Goddard Institute for Space Studies, the average global temperature on Earth "
        "has increased by about 1.1 degrees Celsius (2.0 degrees Fahrenheit) since 1880. "
        "The majority of this warming has occurred since 1975, at a rate of roughly 0.15 to 0.20 degrees Celsius "
        "per decade, driven primarily by human-induced emissions of greenhouse gases."
    ),
    "doc_08_renewable_energy.txt": (
        "Renewable energy sources, such as solar, wind, and hydroelectric power, are critical for reducing carbon emissions. "
        "In 2023, renewable energy accounted for approximately 30% of global electricity generation. "
        "Solar photovoltaic technology has seen the fastest growth due to falling manufacturing costs, "
        "making solar power the cheapest source of new electricity in many parts of the world."
    ),
    "doc_09_battery_chemistry.txt": (
        "Lithium-ion batteries are the dominant technology for portable electronics and electric vehicles. "
        "They rely on the movement of lithium ions between a cathode (usually made of a transition metal oxide) "
        "and an anode (typically graphite) through an electrolyte. The theoretical energy density limit of "
        "conventional lithium-ion batteries is around 350 Wh/kg. Solid-state batteries, which replace the liquid "
        "electrolyte with a solid material, promise higher energy densities and improved safety."
    ),
    "doc_10_hubble_constant.txt": (
        "The Hubble constant (H0) measures the rate at which the universe is expanding. "
        "Measurements of H0 using the cosmic microwave background (Planck satellite) yield a value of "
        "approximately 67.4 km/s/Mpc. However, measurements using local Cepheid variables and supernovae "
        "(Hubble Space Telescope) yield a higher value of approximately 73.0 km/s/Mpc. This persistent "
        "discrepancy, known as the Hubble Tension, suggests potential gaps in our standard model of cosmology."
    ),
    "doc_11_artemis_program.txt": (
        "NASA's Artemis program aims to land the first woman and the next man on the Moon. "
        "The program utilizes the Space Launch System (SLS) rocket and the Orion spacecraft. "
        "Artemis I, an uncrewed test flight around the Moon, was successfully completed in late 2022. "
        "Artemis II will carry a crewed flight path around the Moon, and Artemis III is planned to "
        "land astronauts on the lunar south pole, targeting sustainable lunar exploration."
    ),
    "doc_12_fusion_ignition.txt": (
        "In December 2022, scientists at the Lawrence Livermore National Laboratory's National Ignition Facility (NIF) "
        "achieved fusion ignition for the first time. Fusion ignition occurs when the energy produced by fusion reactions "
        "exceeds the laser energy delivered to the target. The experiment delivered 2.05 megajoules of energy "
        "and yielded 3.15 megajoules of fusion energy, demonstrating a net energy gain (Q > 1)."
    ),
    "doc_13_crispr_gene_editing.txt": (
        "CRISPR-Cas9 is a revolutionary gene-editing technology derived from a bacterial defense mechanism. "
        "It uses a guide RNA sequence to locate a specific DNA sequence and the Cas9 enzyme to cut the DNA at that location. "
        "Once cut, the cell's natural repair mechanisms can be utilized to delete, insert, or modify genes. "
        "This tool has transformed genetic research and holds promise for treating hereditary diseases."
    ),
    "doc_14_photosynthesis_efficiency.txt": (
        "Photosynthesis is the process by which green plants and some other organisms use solar energy to synthesize "
        "nutrients from carbon dioxide and water. The overall thermodynamic efficiency of photosynthesis is relatively low. "
        "Typical crop plants convert only about 1% to 2% of incident sunlight into chemical energy, "
        "with a theoretical upper limit of around 4.5% to 6% for C3 and C4 plants respectively under optimal conditions."
    ),
    "doc_15_dark_matter.txt": (
        "Dark matter is a hypothetical form of matter that does not interact with light or electromagnetic fields, "
        "making it invisible to conventional astronomical instruments. Its existence is inferred from gravitational effects "
        "on visible matter, such as the rotation curves of galaxies and gravitational lensing. "
        "Dark matter is estimated to make up about 85% of the total matter in the universe and 27% of its total energy density."
    ),
    "doc_16_fermi_paradox.txt": (
        "The Fermi Paradox highlights the contradiction between the high probability of extraterrestrial civilizations "
        "existing in the vast universe and the lack of evidence or contact with them. "
        "Proposed solutions include the 'Great Filter' hypothesis (which suggests that intelligent life faces a barrier "
        "preventing its long-term survival), the 'Zoo Hypothesis' (extraterrestrial beings deliberately avoid contacting us), "
        "or simply that the distances and times involved in interstellar communication are too vast."
    ),
    "doc_17_black_hole_entropy.txt": (
        "Bekenstein-Hawking radiation is thermal radiation predicted to be spontaneously emitted by black holes due to "
        "quantum effects near the event horizon. This radiation implies that black holes have entropy and temperature, "
        "and that they will eventually evaporate if they do not absorb mass. "
        "The entropy of a black hole is proportional to the surface area of its event horizon, not its volume, "
        "which led to the formulation of the Holographic Principle."
    ),
    "doc_18_acid_rain.txt": (
        "Acid rain is rain or any other form of precipitation that is unusually acidic, meaning that it has elevated levels "
        "of hydrogen ions (low pH). It is caused by emissions of sulfur dioxide (SO2) and nitrogen oxides (NOx), "
        "which react with water molecules in the atmosphere to produce acids. Acid rain can have harmful effects on "
        "plants, aquatic animals, and infrastructure, particularly those made of marble or limestone."
    ),
    "doc_19_neural_networks_history.txt": (
        "The history of neural networks dates back to the 1940s with the work of Warren McCulloch and Walter Pitts. "
        "In 1958, Frank Rosenblatt invented the Perceptron, which could learn to classify simple patterns. "
        "However, interest waned after Marvin Minsky and Seymour Papert published a book in 1969 proving that single-layer "
        "perceptrons could not solve non-linearly separable problems like XOR. "
        "The field was revitalized in the 1980s with the popularization of the backpropagation training algorithm."
    ),
    "doc_20_graphene_properties.txt": (
        "Graphene is an allotrope of carbon consisting of a single layer of atoms arranged in a two-dimensional honeycomb lattice. "
        "It possesses remarkable properties, including an extremely high tensile strength, excellent electrical and thermal conductivity, "
        "and high optical transparency. Since its isolation in 2004 by Andre Geim and Konstantin Novoselov, "
        "graphene has been researched for applications in electronics, energy storage, and composite materials."
    ),
    "doc_21_solid_state_drives.txt": (
        "Solid-state drives (SSDs) store data using NAND flash memory, which retains data even when power is lost. "
        "Unlike hard disk drives (HDDs), SSDs have no moving parts, resulting in much faster read/write speeds, lower latency, "
        "and higher durability. Modern SSDs utilize NVMe (Non-Volatile Memory Express) protocols over PCIe interfaces to achieve "
        "transfer speeds exceeding 7,000 MB/s, whereas traditional SATA SSDs are capped at around 550 MB/s."
    ),
    "doc_22_kepler_exoplanets.txt": (
        "NASA's Kepler Space Telescope, launched in 2009, revolutionized the study of exoplanets. "
        "It used the transit method, monitoring the brightness of over 150,000 stars to detect periodic dips caused "
        "by planets passing in front of them. Over its nine-year mission, Kepler discovered more than 2,600 confirmed "
        "exoplanets, demonstrating that planets are common throughout the Milky Way and that many stars host earth-sized planets "
        "in their habitable zones."
    ),
    "doc_23_ocean_acidification.txt": (
        "Ocean acidification is the ongoing decrease in the pH of the Earth's oceans, caused by the uptake of carbon dioxide "
        "(CO2) from the atmosphere. When CO2 dissolves in seawater, it forms carbonic acid, which reduces the concentration "
        "of carbonate ions. This makes it more difficult for calcifying organisms, such as corals and shellfish, "
        "to build and maintain their shells and skeletons, threatening marine ecosystems."
    ),
    "doc_24_standard_model_physics.txt": (
        "The Standard Model of particle physics is the theory describing three of the four known fundamental forces "
        "in the universe: electromagnetism, the weak force, and the strong force, while excluding gravity. "
        "It classifies all known elementary particles into quarks, leptons, gauge bosons, and the Higgs boson. "
        "The Higgs boson, discovered at CERN in 2012, is responsible for giving mass to other fundamental particles."
    ),
    "doc_25_gps_relativity.txt": (
        "The Global Positioning System (GPS) relies on precise timing from atomic clocks on satellites. "
        "Because the satellites are in motion and located high above Earth's surface, relativistic effects must be accounted for. "
        "Special relativity dictates that the satellite clocks run slower due to their speed, while general relativity dictates "
        "that they run faster due to the weaker gravitational field. Combined, the satellite clocks run about 38 microseconds "
        "faster per day than clocks on the ground. Without correction, GPS locations would drift by miles within a single day."
    )
}

def generate_corpus():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in DOCUMENTS.items():
        filepath = CORPUS_DIR / filename
        filepath.write_text(content, encoding="utf-8")
        print(f"Generated: {filepath}")

if __name__ == "__main__":
    generate_corpus()
