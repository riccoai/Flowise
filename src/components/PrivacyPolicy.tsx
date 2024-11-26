import { useNavigate } from 'react-router-dom';
import { useEffect } from 'react';
import { ArrowLeft } from 'lucide-react';

const PrivacyPolicy: React.FC = () => {
  const navigate = useNavigate();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const handleBackgroundClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      navigate('/');
    }
  };

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 z-50 overflow-y-auto"
      onClick={handleBackgroundClick}
    >
      <div className="min-h-screen bg-dark p-8 md:p-12 max-w-4xl mx-auto my-8 rounded-lg">
        <button 
          onClick={() => navigate('/')}
          className="inline-flex items-center text-primary hover:text-primary-dark mb-6 transition-colors duration-300 cursor-pointer"
        >
          <ArrowLeft className="w-5 h-5 mr-2" />
          Back to Home
        </button>

        <h1 className="text-4xl font-bold text-white mb-8">Privacy Policy</h1>
        
        <div className="prose prose-invert">
          <p className="text-gray-300">Last updated: {new Date().toLocaleDateString()}</p>
          
          <section className="mt-8">
            <h2 className="text-2xl font-semibold text-white mb-4">1. Information We Collect</h2>
            <p className="text-gray-300">We collect information that you provide directly to us, including:</p>
            <ul className="list-disc pl-6 text-gray-300">
              <li>Name and contact information</li>
              <li>Communication preferences</li>
              <li>Information about your business needs</li>
            </ul>
          </section>

          <section className="mt-8">
            <h2 className="text-2xl font-semibold text-white mb-4">2. How We Use Your Information</h2>
            <p className="text-gray-300">We use the information we collect to:</p>
            <ul className="list-disc pl-6 text-gray-300">
              <li>Provide and improve our services</li>
              <li>Communicate with you about our services</li>
              <li>Respond to your inquiries</li>
            </ul>
          </section>

          <section className="mt-8">
            <h2 className="text-2xl font-semibold text-white mb-4">3. Information Sharing</h2>
            <p className="text-gray-300">
              We do not sell or share your personal information with third parties except as described in this policy.
            </p>
          </section>

          <section className="mt-8">
            <h2 className="text-2xl font-semibold text-white mb-4">4. Contact Us</h2>
            <p className="text-gray-300">
              If you have any questions about this Privacy Policy, please contact us at privacy@ricco.ai
            </p>
          </section>
        </div>
      </div>
    </div>
  );
};

export default PrivacyPolicy; 