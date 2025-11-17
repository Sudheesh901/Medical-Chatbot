from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


DB_USER= "ragsudheesh"
DB_PASS= "Sudhi%40901"  # encoded @ as %40
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "medibot"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine=create_engine(DATABASE_URL)
SessionLocal=sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()
