def _normalize_results(results: dict) -> dict:
    """
    Normalize ChromaDB results.
    Chroma returns lists of lists (batch format). We take the first list.
    """
    if not results or not results['ids']:
        return {
            'ids': [],
            'documents': [],
            'metadatas': [],
            'distances': []
        }
    
    return {
        'ids': results['ids'][0] if results['ids'] else [],
        'documents': results['documents'][0] if results.get('documents') else [],
        'metadatas': results['metadatas'][0] if results.get('metadatas') else [],
        'distances': results['distances'][0] if results.get('distances') else []
    }

def retrieve_router(vm, user_request: str, k_epi=10, k_sem=10, k_skill=3) -> dict:
    """
    Query all memory collections and return normalized results.
    """
    episodic_raw = vm.query_episodic(user_request, k=k_epi)
    semantic_raw = vm.query_semantic(user_request, k=k_sem)
    procedural_raw = vm.query_procedural(user_request, k=k_skill)

    return {
        'episodic': _normalize_results(episodic_raw),
        'semantic': _normalize_results(semantic_raw),
        'procedural': _normalize_results(procedural_raw)
    }
