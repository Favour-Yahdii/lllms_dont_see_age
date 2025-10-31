def generate_summary(text):
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
