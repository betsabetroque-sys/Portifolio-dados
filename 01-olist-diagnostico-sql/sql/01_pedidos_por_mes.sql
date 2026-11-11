-- Pergunta: quantos pedidos por mês?
SELECT strftime('%Y-%m', order_purchase_timestamp) AS mes,
       COUNT(*) AS pedidos
FROM orders
GROUP BY mes
ORDER BY mes;
