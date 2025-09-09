from pymilvus import connections, utility

def cleanup_all_collections():
    # Connect to Milvus
    connections.connect(alias="default", host="localhost", port="19530")

    # List all collections
    collections = utility.list_collections()
    if not collections:
        print("✅ No collections found. Milvus is already clean.")
        return

    print("🗑 Dropping collections...")
    for coll in collections:
        utility.drop_collection(coll)
        print(f"   - Dropped: {coll}")

    # Verify cleanup
    remaining = utility.list_collections()
    if not remaining:
        print("✅ Cleanup complete. No collections remain.")
    else:
        print("⚠️ Some collections still exist:", remaining)
        
def cleanup_alias():
    for alias, conn in connections.list_connections():
        connections.disconnect(alias=alias)
        print(f"🔌 Disconnected alias: {alias}")


if __name__ == "__main__":
    cleanup_all_collections()
    cleanup_alias()
