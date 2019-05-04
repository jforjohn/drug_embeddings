def CRF_get_tag(row):
    """
        Utility method for extracting the gold standard class of entities
    """
    tags = []
    for token in row.tokens:
        start_f = token['char_offset'][0]
        end_f = token['char_offset'][1]
        tag = 'O'
        for parsed_drug in row.parsed_drugs:
            start_d = parsed_drug.offsets[0]
            end_d = parsed_drug.offsets[0]
            if start_f == start_d and end_f <= end_d:
                tag = "B-" + parsed_drug.type
                break
            elif start_f >= start_d and end_f <= end_d:
                tag = "I-" + parsed_drug.type
                break
            '''
            if len(parsed_drug.offsets) > 1:
                for offset in parsed_drug.offsets[1:]:
                    start_d = offset[0]
                    end_d = offset[1]
                    if start_f >= start_d and end_f <= end_d:
                        tag = "I-" + parsed_drug.type
                        break
            '''

        tags.append(tag)

    return tags
