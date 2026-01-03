import yaml

def episode_canonical(e: dict) -> str:
    """Build canonical text for an episode."""
    lines = []
    if e.get('topic'):
        lines.append(f"Topic: {e['topic']}")
    if e.get('title'):
        lines.append(f"Title: {e['title']}")
    if e.get('url'):
        lines.append(f"URL: {e['url']}")
    if e.get('notes'):
        lines.append(f"Notes: {e['notes']}")
    if e.get('tags'):
        lines.append(f"Tags: {e['tags']}")
    if e.get('outcome'):
        lines.append(f"Outcome: {e['outcome']}")
    return "\n".join(lines)

def fact_canonical(f: dict) -> str:
    """Build canonical text for a fact."""
    topic = f.get('topic', '')
    subject = f.get('subject', '')
    predicate = f.get('predicate', '')
    object_ = f.get('object', '')
    return f"{topic} Fact: {subject} {predicate} {object_}"

def skill_canonical(s: dict) -> str:
    """Build canonical text for a skill."""
    name = s.get('name', 'Unknown Skill')
    description = s.get('description', '')
    triggers = ", ".join(s.get('triggers', []))
    steps = " | ".join(s.get('steps', []))
    guardrails = " | ".join(s.get('guardrails', []))
    
    return f"Skill: {name}\nDescription: {description}\nTriggers: {triggers}\nSteps: {steps}\nGuardrails: {guardrails}"

def load_skills(path="skills/skills.yaml") -> list[dict]:
    """Load and validate skills from a YAML file."""
    try:
        with open(path, 'r') as f:
            skills = yaml.safe_load(f)
    except FileNotFoundError:
        return []

    if not isinstance(skills, list):
        print(f"Warning: Skills file {path} must contain a list.")
        return []

    valid_skills = []
    for s in skills:
        if 'id' not in s:
            print(f"Warning: Skill missing 'id', skipping: {s.get('name', 'Unknown')}")
            continue
        valid_skills.append(s)
        
    return valid_skills
