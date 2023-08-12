-- Query 1: Creating a model
create or replace MODEL `sap-ds-demo.sap_churn.churn_bqml_logistic`
OPTIONS (model_type = 'logistic_reg', input_label_cols = ['churn']) AS
SELECT
   *
from
   `sap-ds-demo.sap_churn.churn_indicator_trainingset`;

-- Query 2: performing evaluation
SELECT
   *
FROM
   ML.EVALUATE(MODEL `sap-ds-demo.sap_churn.churn_bqml_logistic`,
   (
      SELECT
         *
      from
         `sap-ds-demo.sap_churn.churn_indicator_test`
   )
)