
import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

function Signup() {
  const [form, setForm] = useState({
    name: '',
    email: '',
    password: ''
  });

  const [message, setMessage] = useState('');
  const navigate = useNavigate(); 

  async function handleSubmit(e) {
    e.preventDefault();

    try {
      const response = await fetch('http://127.0.0.1:8000/register', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(form)
      });

      const data = await response.json();

      if (response.ok) {
        setMessage('Signup successful! Redirecting to login...');
        setTimeout(() => navigate('/login'), 1500); 
      } else {
        setMessage(data.detail || 'Signup failed');
      }

    } catch (err) {
      setMessage('Error connecting to server');
    }
  }

  return (
    <div>
      <h2>Signup</h2>
      <form onSubmit={handleSubmit}>
        <input type="text"
               placeholder="Name"
               onChange={(e) => setForm({ ...form, name: e.target.value })}
        />
        <input type="email"
               placeholder="Email"
               onChange={(e) => setForm({ ...form, email: e.target.value })}
        />
        <input type="password"
               placeholder="Password"
               onChange={(e) => setForm({ ...form, password: e.target.value })}
        />

        <button type="submit">Signup</button>
      </form>
      <p>{message}</p>

      <div>
        Already have an account? <Link to="/login">Login here</Link>
      </div>
    </div>
  );
}

export default Signup;
