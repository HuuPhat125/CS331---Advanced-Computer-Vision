# https://github.com/lupantech/MathVista

# pids: 799, 681, 615
shot_examples = []


def refine_caption(caption):
    if isinstance(caption, str):
        nonsense = ["Sure. ",
                    "Sure, I can do that.",
                    "Sorry, I can't help with images of people yet.",
                    "I can't process this file.",
                    "I'm unable to help you with that, as I'm only a language model and don't have the necessary information or abilities.",
                    "I'm not programmed to assist with that.",
                    "Please let me know if you have any other questions.",
                    "I hope this is helpful!",
                    "I hope this helps!"]
        for non in nonsense:
            caption = caption.replace(non, "").strip()
        caption = caption.replace("  ", " ").strip()
    else:
        caption = ""
    return caption


def refine_ocr(ocr):
    """
    [   (
        [161, 39], [766, 39], [766, 120], [161, 120]],
        'The spring force does',
        0.912845069753024
        ),
    ]
    -->
    [   (
        [161, 39],
        'The spring force does',
        ),
    ]
    """
    try:
        ocr = eval(ocr)
        if len(ocr) > 0:
            ocr = [([int(e[0][0][0]), int(e[0][0][1])], e[1]) for e in ocr]
            ocr = str(ocr)
        else:
            ocr = ""
    except:
        ocr = ""
    return ocr


def create_one_query(problem, examples, shot_num, use_caption, use_ocr, use_context, has_image):


    ### [1] Demo prompt
    if shot_num == 0:
        demo_prompt = ""
    else:
        demos = []
        shot_num = min(shot_num, len(examples))
        for example in examples[:shot_num]:
            prompt = ""

            # question
            prompt += f"Câu hỏi: {example['question']}"

            # choices
            if "options" in example:
                texts = ["Các lựa chọn:"]
                # for i, choice in enumerate(example['options']):
                choices = (example['options'].strip("[]")).split(", '")
                for choice in choices:
                    texts.append(choice.strip("''"))
                    # texts.append(f"({chr(ord('A')+i)}) {choice}")
                prompt += "\n" + "\n".join(texts)

            # caption
            if use_caption:
                caption = example['caption'] if 'caption' in example else ""
                if caption != "":
                    prompt += "\n" + f"Mô tả bức ảnh: {caption}"

            # ocr
            if use_ocr:
                ocr = example['ocr'] if 'ocr' in example else ""
                if ocr != "":
                    prompt += "\n" + f"Nội dung bức ảnh: {ocr}"

            demos.append(prompt)

        demo_prompt = "\n\n".join(demos)

    ### [2] Test query
    # problem info
    question = problem['question']
    caption = problem['caption']
    ocr = problem['ocr']
    context = problem['context']

    # question
    question_text = f"Câu hỏi: {question}"

    # Choices
    choices = []
    for key in ['option_A', 'option_B', 'option_C', 'option_D']:
        if key in problem:
            choices.append(problem[key])

    if choices:
        choices_text = "Các lựa chọn:\n" + "\n".join(choices)
    else:
        choices_text = ""

    # caption
    caption_text = ""
    if use_caption and caption != "":
        caption_text = f"Mô tả của bức ảnh: {caption}"

   # ocr
    ocr_text = ""
    if use_ocr and ocr != "":
        ocr_text = f"Nội dung trên bức ảnh: {ocr}"
    
    #context
    context_text = ""
    if use_context and context != "":
        context_text = f'Nội dung tham khảo:\n {context}'
    if has_image:
        end = "Hãy chọn đáp án đúng bằng Tiếng Việt"
    else:
        end = "Đáp án:"
    elements = [question_text, choices_text, caption_text, ocr_text, context_text, end]
    test_query = "\n".join([e for e in elements if e != ""])

    ### [3] Final query
    query = demo_prompt + "\n\n" + test_query
    query = query.strip()
    return query


def create_query_data(data, args, caption_data, ocr_data, context_data):
    query_data = {}

    for d in data:
        pid = d
        problem = data[d]
        if pid in caption_data:
            caption = caption_data[pid]
            caption = refine_caption(caption)
            problem['caption'] = caption
        else:
            problem['caption'] = ""

        if pid in ocr_data:
            ocr = ocr_data[pid]
            ocr = refine_ocr(ocr)
            problem['ocr'] = ocr
        else:
            problem['ocr'] = []

        if pid in context_data:
            context = context_data[pid]
            problem['context'] = context
        else:
            problem['context'] = ""

        query = create_one_query(
            problem = problem,
            examples = shot_examples,
            shot_num = args.shot_num,
            use_caption = args.use_caption,
            use_ocr = args.use_ocr,
            use_context = args.use_context,
            has_image=args.has_image
            )
        # print(query)
        query_data[pid] = query

    return query_data
