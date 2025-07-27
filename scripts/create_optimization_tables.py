#!/usr/bin/env python3
"""
Create optimization tables in the database.

This script creates the missing optimization tables required for the optimization API.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.connection import db
import structlog

logger = structlog.get_logger()

async def create_optimization_tables():
    """Create optimization tables in the database"""
    
    logger.info("Creating optimization tables...")
    
    # Read the schema file
    schema_path = Path(__file__).parent.parent / "database" / "schema.sql"
    
    if not schema_path.exists():
        logger.error("Schema file not found", path=str(schema_path))
        return False
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    try:
        # Initialize database connection
        await db.initialize()
        logger.info("Connected to database")
        
        # Execute the schema (this will create tables if they don't exist)
        await db.execute(schema_sql)
        logger.info("Schema executed successfully")
        
        # Verify optimization tables were created
        tables_to_check = [
            'optimization_runs',
            'parameter_evaluations', 
            'validation_results',
            'model_registry'
        ]
        
        for table in tables_to_check:
            result = await db.fetch_one(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                table
            )
            
            if result and result['exists']:
                logger.info("Table created successfully", table=table)
            else:
                logger.error("Table not found after creation", table=table)
                return False
        
        logger.info("All optimization tables created successfully!")
        return True
        
    except Exception as e:
        logger.error("Failed to create optimization tables", error=str(e))
        return False
    
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(create_optimization_tables())
