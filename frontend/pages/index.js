
import React, { useState } from 'react';

export default function Home() {
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleClick = async () => {
    setLoading(true);
    const res = await fetch('/api/run_model');
    const data = await res.json();
    setResponse(JSON.stringify(data, null, 2));
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1>ETH Signal Predictor</h1>
      <button onClick={handleClick} disabled={loading}>
        {loading ? 'Running...' : 'Run Model'}
      </button>
      <pre style={{ backgroundColor: '#eee', padding: '10px', marginTop: '20px' }}>
        {response}
      </pre>
    </div>
  );
}
