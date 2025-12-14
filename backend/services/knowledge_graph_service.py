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
    
    async def generate_filtered_graph(self, file_paths: List[str]) -> Dict[str, Any]:
        """Generate knowledge graph data only for specified file paths and their connections."""
        try:
            if not self.vector_store.client:
                await self.vector_store.connect()
            
            collection = self.vector_store.client.collections.get(self.vector_store.collection_name)
            
            # Normalize file paths - extract just the filename
            normalized_paths = set()
            for path in file_paths:
                if "/" in path:
                    normalized_paths.add(path.split("/")[-1])
                elif "\\" in path:
                    normalized_paths.add(path.split("\\")[-1])
                else:
                    normalized_paths.add(path)
            
            if not normalized_paths:
                return {"nodes": [], "links": [], "categories": []}
            
            # Get all documents in batches
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
            
            if not all_objects:
                return {"nodes": [], "links": [], "categories": []}
            
            # Aggregate chunks into files, but only for files matching our filter
            files_map: Dict[str, Dict] = {}
            related_files: Set[str] = set()  # Files that are connected to our filtered files
            
            for obj in all_objects:
                props = obj.properties
                fname = props.get("file_path", "Unknown")
                # Extract filename from path
                if "/" in fname:
                    fname = fname.split("/")[-1]
                elif "\\" in fname:
                    fname = fname.split("\\")[-1]
                
                # Check if this file is in our filter list
                is_target_file = fname in normalized_paths
                
                if is_target_file:
                    if fname not in files_map:
                        files_map[fname] = {
                            "author": props.get("author", "Unknown"),
                            "created_at": props.get("created_at", "Unknown"),
                            "keywords": set(),
                            "chunk_count": 0,
                            "is_target": True
                        }
                    
                    # Process keywords
                    keywords = props.get("keywords", [])
                    if isinstance(keywords, list):
                        for keyword in keywords:
                            if keyword:
                                raw_phrases = [k.strip().lower() for k in str(keyword).split(',') if k.strip()]
                                for phrase in raw_phrases:
                                    words = re.findall(r'\w+', phrase)
                                    valid_words = [w for w in words if len(w) > 2]
                                    files_map[fname]['keywords'].update(valid_words)
                    
                    files_map[fname]['chunk_count'] += 1
            
            # Now find related files (files that share keywords with target files)
            target_keywords: Set[str] = set()
            for fname, info in files_map.items():
                target_keywords.update(info['keywords'])
            
            # Find files that share keywords with target files
            for obj in all_objects:
                props = obj.properties
                fname = props.get("file_path", "Unknown")
                if "/" in fname:
                    fname = fname.split("/")[-1]
                elif "\\" in fname:
                    fname = fname.split("\\")[-1]
                
                # Skip if already in files_map
                if fname in files_map:
                    continue
                
                # Check if this file shares keywords with target files
                keywords = props.get("keywords", [])
                file_keywords = set()
                if isinstance(keywords, list):
                    for keyword in keywords:
                        if keyword:
                            raw_phrases = [k.strip().lower() for k in str(keyword).split(',') if k.strip()]
                            for phrase in raw_phrases:
                                words = re.findall(r'\w+', phrase)
                                valid_words = [w for w in words if len(w) > 2]
                                file_keywords.update(valid_words)
                
                # If shares keywords, add to related files
                if file_keywords.intersection(target_keywords):
                    related_files.add(fname)
                    if fname not in files_map:
                        files_map[fname] = {
                            "author": props.get("author", "Unknown"),
                            "created_at": props.get("created_at", "Unknown"),
                            "keywords": file_keywords,
                            "chunk_count": 0,
                            "is_target": False
                        }
                    files_map[fname]['chunk_count'] += 1
            
            # Build nodes (similar to generate_graph but only for filtered files)
            nodes = []
            categories = []
            author_map = {}
            file_names = list(files_map.keys())
            file_indices = {name: i for i, name in enumerate(file_names)}
            
            current_time = datetime.now()
            
            for fname in file_names:
                info = files_map[fname]
                raw_author = str(info['author']).strip()
                
                is_unknown = raw_author.lower() in ["unknown author", "unknown", "n/a", "", "none"]
                author_key = "Unknown" if is_unknown else raw_author
                
                if author_key not in author_map:
                    author_map[author_key] = len(categories)
                    categories.append({"name": author_key})
                
                # Calculate opacity
                opacity = 1.0
                try:
                    created_dt = None
                    date_str = str(info['created_at']).strip()
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
                
                # Target files get higher opacity and different color
                item_style = {
                    "opacity": opacity if info['is_target'] else opacity * 0.6,
                    "borderColor": "#001158" if info['is_target'] else ("#fff" if opacity > 0.8 else "transparent"),
                    "borderWidth": 2 if info['is_target'] else 1
                }
                label_style = {"show": True, "color": "#001158" if info['is_target'] else ("#333" if opacity > 0.7 else "#aaa")}
                
                if is_unknown:
                    item_style["color"] = "#cccccc"
                    label_style["color"] = "#999999"
                elif info['is_target']:
                    item_style["color"] = "#CAE9FF"  # Highlight target files
                
                nodes.append({
                    "id": str(file_indices[fname]),
                    "name": fname,
                    "category": author_map[author_key],
                    "value": info['chunk_count'],
                    "itemStyle": item_style,
                    "symbolSize": 20 if info['is_target'] else 15,
                    "label": label_style
                })
            
            # Build edges (links) - only between files in our filtered set
            links = []
            connection_counts = {i: 0 for i in range(len(nodes))}
            
            for i in range(len(file_names)):
                for j in range(i + 1, len(file_names)):
                    source_file = file_names[i]
                    target_file = file_names[j]
                    
                    kw_a = files_map[source_file]['keywords']
                    kw_b = files_map[target_file]['keywords']
                    
                    intersection = kw_a.intersection(kw_b)
                    match_count = len(intersection)
                    
                    if match_count > 0:
                        width = 1 + (match_count * 0.5)
                        # Highlight links between target files
                        is_target_link = files_map[source_file]['is_target'] and files_map[target_file]['is_target']
                        links.append({
                            "source": str(i),
                            "target": str(j),
                            "value": match_count,
                            "lineStyle": {
                                "width": min(width, 8),
                                "type": "solid" if is_target_link else "dashed",
                                "opacity": 0.8 if is_target_link else 0.4,
                                "curveness": 0.2,
                                "color": "#001158" if is_target_link else "source"
                            },
                            "tooltip": {"formatter": f"Shared: {', '.join(list(intersection)[:10])}"}
                        })
                        connection_counts[i] += 1
                        connection_counts[j] += 1
            
            # Adjust node size based on connection count
            for i in range(len(nodes)):
                base_size = 20 if files_map[file_names[i]]['is_target'] else 15
                nodes[i]["symbolSize"] = min(base_size + (connection_counts[i] * 2), 70)
            
            return {
                "nodes": nodes,
                "links": links,
                "categories": categories
            }
            
        except Exception as e:
            print(f"Error generating filtered knowledge graph: {e}")
            import traceback
            traceback.print_exc()
            return {"nodes": [], "links": [], "categories": []}
