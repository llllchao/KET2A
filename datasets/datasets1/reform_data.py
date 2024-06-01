import re
"""
    raw.txt:
        encoding: utf-8
        description: raw datasets
    entities.dict:
        encoding: utf-8
        description: nodes in triplets
    relations.dict:
        encoding: utf-8
        description: relations in triplets
    triplets：
        encoding: utf-8
        description: all triplets
    out-entities.txt: (no use)
        encoding: utf-8
        description: output entities in output sequences
    in-entities.txt: (no use)
        encoding: utf-8
        description: input entities in input sequences
    all.txt:
        encoding: utf-8
        description: sentence pairs
"""

part_in_area_file = "raw.txt"

with open(part_in_area_file, 'r', encoding='utf-8') as fr:
    triplets = []
    relations = []
    entities = []
    actions = []
    commands_and_sequences = {}

    # tmp variables
    sequence = []  # action seq
    command = []  # area、item、target、rule

    for line in fr.readlines():
        triplet = re.split(r'\s+', line.strip())

        if len(triplet) != 3:
            continue

        # the beginning of a new template
        if triplet[1] == "in":
            if len(sequence) != 0:
                command_seq = ' '.join(command)
                action_seq = ' '.join(sequence)
                commands_and_sequences[command_seq] = action_seq

                command.clear()
                sequence.clear()

            command.append(triplet[2])
            command.append(triplet[0])
            continue

        if triplet[1] == 'action' or triplet[1] == 'manipulation':
            action = triplet[2]
            sequence.append(action)
            if action not in actions:
                actions.append(action)
            continue  # do not add to knowledge graph

        if triplet[1] == 'rule':
            command.append(triplet[2])
            continue  # do not add to knowledge graph

        if triplet[1] == 'target':
            command.append(triplet[2])
            continue  # do not add to knowledge graph

        # if triplet[1] == 'parameter':
        #     continue  # do not add to knowledge graph

        # if triplet[1] == 'function':
        #     continue  # do not add to knowledge graph

        # if triplet[1] == 'description':
        #     continue  # do not add to knowledge graph

        if triplet not in triplets:
            triplets.append(triplet)

        if triplet[1] not in relations:
            relations.append(triplet[1])

        if triplet[0] not in entities:
            entities.append(triplet[0])

        if triplet[2] not in entities:
            entities.append(triplet[2])

print(triplets)
print(relations)
print(entities)
print(actions)
print(commands_and_sequences)

# save to files
with open('graph/triplets.txt', 'w', encoding='utf-8') as fw:
    for triplet in triplets:
        fw.write('\t'.join(triplet))
        fw.write('\n')

with open('graph/relations.dict', 'w', encoding='utf-8') as fw:
    count = 0
    for r in relations:
        fw.write(str(count) + '\t' + r)
        fw.write('\n')
        count += 1

with open('graph/entities.dict', 'w', encoding='utf-8') as fw:
    count = 0
    for e in entities:
        fw.write(str(count) + '\t' + e)
        fw.write('\n')
        count += 1

with open('seq2seq/in-entities.txt', 'w', encoding='utf-8') as fw:
    seq_in_entities = []
    for c in commands_and_sequences:
        for e in re.split(r"\s+", c):
            if e not in seq_in_entities:
                seq_in_entities.append(e)

    count = 0
    for e in seq_in_entities:
        fw.write(str(count) + '\t' + e)
        fw.write('\n')
        count += 1

with open('seq2seq/out-entities.txt', 'w', encoding='utf-8') as fw:
    count = 0
    for e in actions:
        fw.write(str(count) + '\t' + e)
        fw.write('\n')
        count += 1

with open('seq2seq/all.txt', 'w', encoding='utf-8') as fw:
    for c, seq in commands_and_sequences.items():
        fw.write(c + '\t' + seq)
        fw.write('\n')
