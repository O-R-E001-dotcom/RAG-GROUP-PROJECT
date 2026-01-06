
import { useState } from "react";
import { useNavigate , Link} from "react-router-dom";

export default function Login({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  async function handleLogin(e) {
    e.preventDefault();
    try{
      const res = await fetch("http://localhost:8000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
     
      if (res.ok) {
        setMessage("Login successful!");
        localStorage.setItem("token", data.access_token); 
        onLogin(); 
        navigate("/chatbox"); 
      } else {
        setMessage(data.detail || 'Invalid credentials');
      }

    } catch (err) {
      setMessage("Error connecting to server");
    }
  }
  

  return (
    <div className="p-6 max-w-sm mx-auto">
      <form onSubmit={handleLogin}>
        <input
        placeholder="Email"
        onChange={(e) => setEmail(e.target.value)}
        className="border p-2 w-full mb-2"
        />
        <input
          type="password"
          placeholder="Password"
          onChange={(e) => setPassword(e.target.value)}
          className="border p-2 w-full mb-2"
        />
        <button onClick={submit} className="bg-green-700 text-white p-2 w-full">
          Login
        </button>
      </form>
      <div>
        Don't have an account? <Link to="/Signup">Sign up</Link>
      </div>
    </div>
  );
}



  






