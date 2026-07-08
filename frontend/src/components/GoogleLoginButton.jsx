import { useAuth } from '../contexts/AuthContext';
import { useState } from 'react';

export const GoogleLoginButton = () => {
  const { login } = useAuth();
  const [error, setError] = useState('');

  const handleMockLogin = async (role) => {
    setError('');
    const success = await login(`mock_${role}@university.edu`);
    if (!success) {
      setError('Login failed');
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <button 
        onClick={() => handleMockLogin('teacher')}
        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
      >
        Login as Teacher (Mock)
      </button>
      <button 
        onClick={() => handleMockLogin('student')}
        className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
      >
        Login as Student (Mock)
      </button>
      {error && <p className="text-red-500 text-sm">{error}</p>}
    </div>
  );
};
