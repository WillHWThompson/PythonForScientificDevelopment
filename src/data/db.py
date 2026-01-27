from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel, create_engine, Session, select

# SQLModel combines Pydantic and SQLAlchemy
# Perfect for scientific developers who want type-safety + SQL

class ExperimentResult(SQLModel, table=True):
    """
    ORM Model for tracking experiment outcomes.
    Acts as both a Pydantic schema and a SQL table definition.
    """
    experiment_name: str = Field(primary_key=True, description="Unique name of the experiment.")
    accuracy: float
    f1_score: float
    timestamp: datetime = Field(default_factory=datetime.now)
    parameters: str # JSON string of hyperparameters

# DuckDB connection string
DATABASE_URL = "duckdb:///results.db"
engine = create_engine(DATABASE_URL)

def init_db():
    """Create the database and tables."""
    SQLModel.metadata.create_all(engine)
    print("DuckDB database initialized with SQLModel!")

def save_result(name: str, acc: float, f1: float, params: str):
    """Save an experiment result to the database."""
    result = ExperimentResult(
        experiment_name=name,
        accuracy=acc,
        f1_score=f1,
        parameters=params
    )
    with Session(engine) as session:
        session.add(result)
        session.commit()
    print(f"Saved result for experiment: {name}")

def get_latest_results(limit: int = 5):
    """Retrieve the most recent results."""
    with Session(engine) as session:
        statement = select(ExperimentResult).order_by(ExperimentResult.timestamp.desc()).limit(limit)
        results = session.exec(statement).all()
        return results

if __name__ == "__main__":
    # Quick demonstration
    init_db()
    save_result("bert_sweep_01", 0.92, 0.91, '{"lr": 5e-5, "adim": 256}')
    save_result("bert_sweep_02", 0.94, 0.93, '{"lr": 3e-5, "adim": 512}')
    
    print("\nLatest Results:")
    for r in get_latest_results():
        print(f"[{r.timestamp}] {r.experiment_name} - Acc: {r.accuracy:.4f}")
