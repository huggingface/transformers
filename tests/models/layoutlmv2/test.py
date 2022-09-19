# for example in range(self.batch_size):
            #     # sample a number of entities for the example
            #     num_entities = random.randint(1, self.max_entities)
            #     entity_starts = []
            #     entity_ends = []
            #     entity_labels = []
            #     for entity in range(num_entities):
            #         entity_start = random.randint(0, self.seq_length)
            #         entity_end = entity_start + random.randint(1, self.max_entity_length)
            #         entity_label = random.randint(0, self.num_labels)
            #         entity_starts.append(entity_start)
            #         entity_ends.append(entity_end)
            #         entity_labels.append(entity_label)
            #     entity_dict = {
            #         "start": entity_starts,
            #         "end": entity_ends,
            #         "label": entity_labels,
            #     }
            #     entities.append(entity_dict)

            #     # sample a number of relations for the example
            #     num_relations = random.randint(1, self.max_relations)
            #     start_indices = []
            #     end_indices = []
            #     heads = []
            #     tails = []
            #     for relation in range(num_relations):
            #         start_index = random.randint(0, self.seq_length)
            #         end_index = start_index + random.randint(1, self.max_entity_length)
            #         head = random.randint(0, self.max_entities)
            #         tail = random.randint(0, self.max_entities)
            #     relation_dict = {
            #         "start_index": start_indices,
            #         "end_index": end_indices,
            #         "head": heads,
            #         "tail": tails,
            #     }
            #     relations.append(relation_dict)