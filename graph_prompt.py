AGE_EXTRACTION_SYS_PROMPT = """

    You are given a text consisting of one or more research abstracts. Your task is to extract all unique age-related entities mentioned in the text.

    An age-related entity is any phrase or term that describes the age or life-stage of a population or participant group. This includes (but is not limited to):

    Numerical age ranges or cutoffs (e.g. “aged 18–24”, “children under 5”, “participants older than 65 years”)

    Life-stage or demographic labels (e.g. “adults”, “elderly”, “infants”, “adolescents”, “middle-aged”, “neonates”)

    Relative age comparisons (e.g. “older adults”, “younger participants”)

    Age-related qualifiers in disease or intervention contexts (e.g. “pediatric patients”, “geriatric population”)

    Please extract only these age-related entities, and ensure not to duplicate already seen entities in your extraction.

    Do not provide any additional explanation or commentary. If there are no age-entities present do not return any other entity.
    Each entity should be separated by a comma only, do not add any newline or tab characters."""
    response = client.responses.create(
        instructions=AGE_EXTRACTION_SYS_PROMPT,
        model="gpt-4.1-nano",
        input=f"Here is the abstract set: \n{text}"
    )
    return response.output_text

    AGE_EXTRACTION_SYS_PROMPT = """

    You are given a text consisting of one or more research abstracts. Your task is to extract all unique age-related entities mentioned in the text.

    An age-related entity is any phrase or term that describes the age or life-stage of a population or participant group. This includes (but is not limited to):

    Numerical age ranges or cutoffs (e.g. “aged 18–24”, “children under 5”, “participants older than 65 years”)

    Life-stage or demographic labels (e.g. “adults”, “elderly”, “infants”, “adolescents”, “middle-aged”, “neonates”)

    Relative age comparisons (e.g. “older adults”, “younger participants”)

    Age-related qualifiers in disease or intervention contexts (e.g. “pediatric patients”, “geriatric population”)

    Please extract only these age-related entities, and ensure not to duplicate already seen entities in your extraction.

    Do not provide any additional explanation or commentary. If there are no age-entities present do not return any other entity.
    Each entity should be separated by a comma only, do not add any newline or tab characters."""
    response = client.responses.create(
        instructions=AGE_EXTRACTION_SYS_PROMPT,
        model="gpt-4.1-mini-2025-04-14",
        #temperature=0,
        input=f"Here is the abstract set: \n{text}"
    )
    return response.output_text


  SYS_PROMPT = """You are an experienced and objective biomedical systematic reviewer.

  Your task is to draft a concise, structured abstract that approximates a systematic review abstract, using the set of provided biomedical research abstracts.

  Each abstract is delimited by the text "ABSTRACT"  followed by a number eg "ABSTRACT 1" for the first abstract.

  Identify and summarize the core research objectives, methods, key findings, and conclusions across the studies. Highlight:

  Common research themes or clinical questions,

  Consistencies and discrepancies in results,

  Any notable patterns in patient populations, interventions, or outcomes.

  Write in a clear, professional tone suitable for publication in a biomedical journal. Ensure the summary is informative, evidence-focused, and understandable to clinicians, researchers, and healthcare policymakers.

  You should consider the following structure for your summary:

  Background: What is the common research question or clinical issue?
  Methods Summary: What types of studies or approaches were used across abstracts?
  Findings: What are the major results and points of consensus or conflict?
  Conclusion: What do the studies collectively suggest about the clinical or scientific implications?
  Ensure to produce your summary abstract based only on the provided abstracts.
  Do not include any external information or personal opinions.

  Your summary should be a synthesis of the provided abstracts, not a critique or evaluation of them."""
  response = client.responses.create(
      instructions=SYS_PROMPT,
      model="gpt-4.1-nano", #gpt-4.1-nano #gpt-4.1-mini-2025-04-14
      temperature=0,
      max_output_tokens=750,
      input=f"Here is the abstract set: \n{text}"
  )
  return response.output_text


  SYS_PROMPT = """You are an experienced and objective biomedical systematic reviewer.

  Your task is to draft a concise, structured abstract that approximates a systematic review abstract, using the set of provided biomedical research abstracts.

  The provided research abstracts all involve studies conducted specifically in older adult populations, remember this as you complete the task.

  Each abstract is delimited by the text "ABSTRACT"  followed by a number eg "ABSTRACT 1" for the first abstract.

  Identify and summarize the core research objectives, methods, key findings, and conclusions across the studies. Highlight:

  Common research themes or clinical questions,

  Consistencies and discrepancies in results,

  Any notable patterns in patient populations, interventions, or outcomes.

  Write in a clear, professional tone suitable for publication in a biomedical journal. Ensure the summary is informative, evidence-focused, and understandable to clinicians, researchers, and healthcare policymakers.

  You should consider the following structure for your summary:

  Background: What is the common research question or clinical issue?
  Methods Summary: What types of studies or approaches were used across abstracts?
  Findings: What are the major results and points of consensus or conflict?
  Conclusion: What do the studies collectively suggest about the clinical or scientific implications?
  Ensure to produce your summary abstract based only on the provided abstracts.
  Do not include any external information or personal opinions.

  Your summary should be a synthesis of the provided abstracts, not a critique or evaluation of them."""
  response = client.responses.create(
      instructions=SYS_PROMPT,
      model="gpt-4.1-nano",
      temperature=0,
      max_output_tokens=750,
      input=f"Here is the abstract set: \n{text}"
  )
  return response.output_text

SYS_PROMPT = ("""

You are an intelligent assistant tasked with extracting structured information from academic PDF documents. Please follow the instructions carefully to ensure accuracy and completeness.

Step 1: Extract Key Information

Your goal is to extract the following elements:

- "title": The full title of the study
- "firstauthor": The name of the first author
- "year": The year the study was published
- "abstract": The full abstract of the study
- "systematicreviewpmid": Always use the fixed PubMed ID `"31077009"`
- "stype": The study type or study design (e.g., randomized controlled trial, cohort study, etc.)
- "populationdemographics": Always use `"childhood"`

Important notes for extracting the abstract:
- The abstract may be spread across multiple paragraphs or sections. Be sure to read through the entire document or all text chunks provided.
- Do not stop early — continue extracting until you have reached the end of the abstract.
- Stop extracting the abstract only when you reach the section labeled **"Keywords"** (or a variation like `"Keyword(s):"`).
- If a chunk feels like it contains only part of the abstract, continue with the next chunk until the abstract is complete.

Step 2: Format Your Output

Return your result as a valid JSON object using this exact structure:

{{
  "title": "The title of the study",
  "firstauthor": "The first author of the study",
  "year": "The year the study was published",
  "abstract": "The full abstract of the study",
  "systematicreviewpmid": "31077009",
  "stype": "The study type or study design",
  "populationdemographics": "childhood"
}}


Do **not** return any extra explanation, commentary, or formatting outside of the JSON object.""")


#initialise llm
llm = ChatOllama(model="...", temperature=0)
class Triples(BaseModel):
    title: str = Field(description="The title of the study")

    firstauthor: str = Field(description="The first author of the study")

    year: str = Field(description="The year the study was published")

    abstract: str = Field(description="The abstract of the study")

    systematicreviewpmid: str = Field(description="PubMed ID of the review, this is always '31077009'")

    stype: str = Field(description="The study type or study design")

    populationdemographics: str = Field(description="The population demographics, always write 'childhood'")


parser = JsonOutputParser(pydantic_object=Triples)

def graphPrompt(input_, model=llm, parser=parser):


    prompt = ChatPromptTemplate(
    messages = [
        ("system", SYS_PROMPT),
        ("user", "Extract the relevant information from the following in the format provided: {context}")],
    input_variables = ["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model | parser
    output = chain.invoke({"context": input_})
    return output
