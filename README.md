# final_exam_study_tool
Help you study for your final

## Setup Documentation

### ArcadeDB (Docker Setup)

We use Docker to run ArcadeDB. This requires Docker and Docker Compose to be installed.

1. **Configuration**: Create a `docker-compose.yml` file in the root directory (already provided in the repo):

   ```yaml
   version: '3.8'

   services:
     arcadedb:
       image: ghcr.io/arcadedata/arcadedb:latest
       container_name: arcadedb
       environment:
         - ARCADEDB_ROOT_PASSWORD=securepassword
         - JAVA_OPTS=-Darcadedb.server.rootPassword=securepassword
       ports:
         - "2480:2480" # HTTP
         - "2424:2424" # Binary
       volumes:
         - ./database/ArcadeDB/arcadedb-24.10.1/databases:/home/arcadedb/databases
         - ./database/ArcadeDB/arcadedb-24.10.1/config:/home/arcadedb/config
         - ./database/ArcadeDB/arcadedb-24.10.1/log:/home/arcadedb/log
       restart: unless-stopped
   ```

2. **Start the Server**:
   ```bash
   docker compose up -d
   ```

3. **Verify**:
   The server will be available at [http://localhost:2480](http://localhost:2480).
   
   Login credentials:
   - **User**: `root`
   - **Password**: `securepassword`

   