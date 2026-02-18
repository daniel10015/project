import sqlite3
import pandas as pd

# 1. DB 연결
conn = sqlite3.connect("report_cuda_nvtx.sqlite")

# 2. SQL 쿼리: Runtime 테이블과 StringIds 테이블을 합쳐서 읽기
query = """
SELECT 
    str.value AS FunctionName,     -- 이름 (StringIds에서 가져옴)
    r.start, 
    r.end,
    (r.end - r.start) AS Duration, -- 걸린 시간
    r.correlationId
FROM 
    CUPTI_ACTIVITY_KIND_RUNTIME AS r
JOIN 
    StringIds AS str
ON 
    r.nameId = str.id              -- 암호 해독 (nameId와 id를 매칭)
LIMIT 20                           -- 20개만 보여주기
"""

# 3. 결과 출력
df = pd.read_sql_query(query, conn)
print(df)

conn.close()