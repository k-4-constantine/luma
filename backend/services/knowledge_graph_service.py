"""Knowledge graph service for generating document network visualization."""

import re
from datetime import datetime
from typing import Dict, List, Set, Any
from backend.services.vector_store import VectorStore


class KnowledgeGraphService:
    """Service for generating knowledge graph data from vector store."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize knowledge graph service."""
        self.vector_store = vector_store
    
    async def generate_graph(self) -> Dict[str, Any]:
        """Generate knowledge graph data from Weaviate documents."""
        try:
            if not self.vector_store.client:
                await self.vector_store.connect()
            
            collection = self.vector_store.client.collections.get(self.vector_store.collection_name)
            
            # Get all documents in batches to avoid gRPC message size limit
            # Fetch without vectors first, then get vectors separately if needed
            all_objects = []
            batch_size = 100
            offset = 0
            
            while True:
                response = collection.query.fetch_objects(limit=batch_size, offset=offset, include_vector=False)
                if not response.objects:
                    break
                all_objects.extend(response.objects)
                if len(response.objects) < batch_size:
                    break
                offset += batch_size
            
            # Now fetch vectors separately in smaller batches
            # We'll get vectors only for files we need (one per file)
            if not all_objects:
                return {"nodes": [], "links": [], "categories": []}
            
            # Aggregate chunks into files
            files_map: Dict[str, Dict] = {}
            
            for obj in all_objects:
                props = obj.properties
                fname = props.get("file_path", "Unknown")
                # Extract filename from path
                if "/" in fname:
                    fname = fname.split("/")[-1]
                elif "\\" in fname:
                    fname = fname.split("\\")[-1]
                
                if fname not in files_map:
                    files_map[fname] = {
                        "author": props.get("author", "Unknown"),
                        "created_at": props.get("created_at", "Unknown"),
                        "keywords": set(),
                        "chunk_count": 0
                    }
                
                # Process keywords
                keywords = props.get("keywords", [])
                if isinstance(keywords, list):
                    for keyword in keywords:
                        if keyword:
                            # Tokenize keywords - split by comma and extract words
                            raw_phrases = [k.strip().lower() for k in str(keyword).split(',') if k.strip()]
                            for phrase in raw_phrases:
                                words = re.findall(r'\w+', phrase)
                                valid_words = [w for w in words if len(w) > 2]
                                files_map[fname]['keywords'].update(valid_words)
                
                files_map[fname]['chunk_count'] += 1
            
            # Build nodes
            nodes = []
            categories = []
            author_map = {}
            file_names = list(files_map.keys())
            file_indices = {name: i for i, name in enumerate(file_names)}
            
            current_time = datetime.now()
            
            for fname in file_names:
                info = files_map[fname]
                raw_author = str(info['author']).strip()
                
                # Author cleaning and categorization
                is_unknown = raw_author.lower() in ["unknown author", "unknown", "n/a", "", "none"]
                author_key = "Unknown" if is_unknown else raw_author
                
                if author_key not in author_map:
                    author_map[author_key] = len(categories)
                    categories.append({"name": author_key})
                
                # Time-based opacity calculation
                opacity = 1.0
                try:
                    created_dt = None
                    date_str = str(info['created_at']).strip()
                    # Try parsing multiple date formats
                    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y%m%d"]:
                        try:
                            if 'T' in date_str:
                                created_dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            else:
                                created_dt = datetime.strptime(date_str, fmt)
                            break
                        except:
                            pass
                    
                    if created_dt:
                        days_diff = (current_time - created_dt.replace(tzinfo=None) if created_dt.tzinfo else created_dt).days
                        if days_diff < 0:
                            days_diff = 0
                        if days_diff <= 90:
                            opacity = 1.0
                        elif days_diff <= 365:
                            opacity = 0.8
                        else:
                            opacity = max(0.2, 0.8 - ((days_diff - 365) / (365 * 2)))
                    else:
                        opacity = 0.5
                except:
                    opacity = 0.5
                
                # Node style configuration
                item_style = {
                    "opacity": opacity,
                    "borderColor": "#fff" if opacity > 0.8 else "transparent",
                    "borderWidth": 1
                }
                label_style = {"show": True, "color": "#333" if opacity > 0.7 else "#aaa"}
                
                if is_unknown:
                    item_style["color"] = "#cccccc"
                    label_style["color"] = "#999999"
                
                nodes.append({
                    "id": str(file_indices[fname]),
                    "name": fname,
                    "category": author_map[author_key],
                    "value": info['chunk_count'],
                    "itemStyle": item_style,
                    "symbolSize": 15,
                    "label": label_style
                })
            
            # Build edges (links) - only based on keyword matching for now
            links = []
            connection_counts = {i: 0 for i in range(len(nodes))}
            
            for i in range(len(file_names)):
                for j in range(i + 1, len(file_names)):
                    source_file = file_names[i]
                    target_file = file_names[j]
                    
                    # Keyword intersection matching
                    kw_a = files_map[source_file]['keywords']
                    kw_b = files_map[target_file]['keywords']
                    
                    intersection = kw_a.intersection(kw_b)
                    match_count = len(intersection)
                    
                    # Connect if there's at least 1 common word
                    if match_count > 0:
                        width = 1 + (match_count * 0.5)
                        links.append({
                            "source": str(i),
                            "target": str(j),
                            "value": match_count,
                            "lineStyle": {
                                "width": min(width, 8),
                                "type": "solid",
                                "opacity": 0.6,
                                "curveness": 0.2,
                                "color": "source"
                            },
                            "tooltip": {"formatter": f"Shared: {', '.join(list(intersection)[:10])}"}
                        })
                        connection_counts[i] += 1
                        connection_counts[j] += 1
            
            # Adjust node size based on connection count
            for i in range(len(nodes)):
                nodes[i]["symbolSize"] = min(10 + (connection_counts[i] * 2), 70)
            
            return {
                "nodes": nodes,
                "links": links,
                "categories": categories
            }
            
        except Exception as e:
            print(f"Error generating knowledge graph: {e}")
            import traceback
            traceback.print_exc()
            return {"nodes": [], "links": [], "categories": []}
