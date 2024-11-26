import React from 'react';
import { motion } from 'framer-motion';
import { Sparkles } from 'lucide-react';

const NewsletterSignup: React.FC = () => {
  return (
    <motion.div
      className="border-t border-gray-800 pt-8"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
    >
      <div className="max-w-md mx-auto text-center">
        <div className="flex items-center justify-center mb-4">
          <Sparkles className="w-5 h-5 text-primary mr-2" />
          <h3 className="text-lg font-semibold">AI Pulse</h3>
        </div>
        <p className="text-sm text-gray-400 mb-6">
          Get weekly insights on AI trends, tools, and tips delivered to your inbox.
        </p>
        <form className="flex gap-2">
          <input
            type="email"
            placeholder="Enter your email"
            className="flex-1 px-4 py-2 rounded-lg bg-dark-lighter border border-gray-700 text-white focus:border-primary focus:ring focus:ring-primary focus:ring-opacity-50"
          />
          <motion.button
            type="submit"
            className="px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors duration-300"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Subscribe
          </motion.button>
        </form>
      </div>
    </motion.div>
  );
};

export default NewsletterSignup;