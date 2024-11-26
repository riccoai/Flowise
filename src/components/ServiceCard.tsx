import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';

interface ServiceCardProps {
  id: string;
  icon: React.ReactNode;
  title: string;
  description: string;
  details: {
    features: string[];
    benefits: string[];
    idealFor: string;
    image: string;
  };
  isExpanded: boolean;
  onExpand: (expanded: boolean) => void;
}

const ServiceCard: React.FC<ServiceCardProps> = ({ 
  id, 
  icon, 
  title, 
  description, 
  details,
  isExpanded,
  onExpand
}) => {
  return (
    <motion.div
      className="bg-dark rounded-lg border border-gray-800 overflow-hidden"
      onMouseEnter={() => onExpand(true)}
      onMouseLeave={() => onExpand(false)}
    >
      <div className="p-8">
        <div className="flex justify-between items-start">
          <div className="mb-4">{icon}</div>
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1, rotate: isExpanded ? 180 : 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <ChevronDown className="w-6 h-6 text-primary" />
            </motion.div>
          )}
        </div>
        <h3 className="text-xl font-bold mb-3 hover-glow">{title}</h3>
        <p className="text-gray-400">{description}</p>
      </div>

      <AnimatePresence mode="wait">
        {isExpanded && (
          <motion.div
            key={`${id}-content`}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="border-t border-gray-800"
          >
            <div className="p-8">
              <div className="mb-6">
                <img
                  src={details.image}
                  alt={`${title} illustration`}
                  className="w-full h-48 object-cover rounded-lg mb-6"
                />
              </div>
              
              <div className="space-y-6">
                <div>
                  <h4 className="text-lg font-semibold mb-2 text-primary">Key Features</h4>
                  <ul className="list-disc list-inside text-gray-400 space-y-1">
                    {details.features.map((feature, index) => (
                      <li key={`${id}-feature-${index}`}>{feature}</li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="text-lg font-semibold mb-2 text-primary">Benefits</h4>
                  <ul className="list-disc list-inside text-gray-400 space-y-1">
                    {details.benefits.map((benefit, index) => (
                      <li key={`${id}-benefit-${index}`}>{benefit}</li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="text-lg font-semibold mb-2 text-primary">Ideal For</h4>
                  <p className="text-gray-400">{details.idealFor}</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default ServiceCard;