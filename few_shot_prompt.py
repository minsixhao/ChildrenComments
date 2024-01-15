from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser

# import langchain
# langchain.debug = True
import os
os.environ["OPENAI_API_KEY"]="sk-MSthzgbKSf7cd8v286F9BaBe89Bd4754BbD6Bf108f4aBd05"
os.environ["OPENAI_API_BASE"]="https://oneapi.xty.app/v1"

chat_Model = ChatOpenAI(temperature=0.9, model_name='gpt-3.5-turbo')
llm = OpenAI(temperature=0.6)
def Question_classify():
    examples = [
        {
            "question": "对学习认真的孩子",
            "answer": """
    你是一个对学习认真的孩子，上课时总能举手发言，积极动脑，学习比较出色。希望能积极配合老师参加集体活动，继续努力，多向优秀的同学学习，取长补短。
    """,
        },
        {
            "question": "性格内心的女孩",
            "answer": """
    你是一个性格内向的女孩，上课能专心听讲，作业能按时完成，学习上认真努力，但思维反应较慢希望今后多阅读、多学习，老师期待你的进步!
    """,
        },
        {
            "question": "好动可爱的男孩",
            "answer": """
    你是一个好动、可爱的男孩，能与同学和睦相处学习上有自觉性，作业也能按时完成，希望学习上多用一份心，做一个活泼可爱富有朝气的好少年
    """,
        },
        {
            "question": "乐观的学生",
            "answer": """
    你是一个乐观的学生，课堂上积极举手发言，听从老师的安排，积极参加学校的各项活动。在遵守纪律方面，你做得不错。多多练字，新学期给老师一个惊喜吧。
    """,
        },
    ]

    prefix = """
    你是小学一年级的班主任，现在要根据每一个小朋友性格，回答出他的学期评语。\n下面是一些示例：
    """
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"], template="Question: {question}\n{answer}"
    )

    prompt = FewShotPromptTemplate(
        examples = examples,
        example_prompt = example_prompt,
        prefix=prefix,
        suffix = f"Question:{input}",
        input_variables = ['input']
    )
    chain = LLMChain(llm=chat_Model, prompt=prompt)
    return chain

def generate_children_character():
    prompt_template = """
    根据一年级的小朋友的特点生成简短的描述
    例如：对学习认真的孩子,性格内心的女孩,好动可爱的男孩，乐观的学生。
    描述的结构是:形容词 + 名词,长度不超过十个汉字.
    形容词的特点包括性格、外貌、爱好等方面.
    名词的范围是：男孩，女孩，孩子，学生。
    """
    output_parser = CommaSeparatedListOutputParser()

    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="List 5 {subject}.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions}
    )
    _input = prompt.format(subject = prompt_template)
    output = llm.invoke(_input)
    desc_list = output_parser.parse(output)
    return desc_list

if __name__ == '__main__':

    desc_list =  generate_children_character()
    print(desc_list)

    for idx,input in enumerate(desc_list, 1):
        chain = Question_classify()
        result = chain.invoke({'input': input})
        print('问题{}:\n'.format(idx), result['input'])
        print('答案:\n', result['text'])