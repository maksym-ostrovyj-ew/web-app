/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  Place,
  Destination,
  ItineraryFlowInput,
  ItineraryFlowOutput,
} from './types';

import {
  getActivitiesForDestination,
  placesRetriever,
} from './placesRetriever';
import { ai } from './genkit.config';

import { z } from 'genkit';

export const ItineraryGeneratorPromptInput = ai.defineSchema(
  'ItineraryGeneratorPromptInput',
  z.object({
    request: z.string(),
    place: z.string(),
    placeDescription: z.string(),
    activities: z.array(
      z.object({
        name: z.string(),
        description: z.string(),
        imageUrl: z.string().optional(),
      }),
    ),
  }),
);

const generateItinerary = async (request: string, place: Place) => {
  const activities = await getActivitiesForDestination(place.ref);

  const itineraryGenerator = await ai.prompt<
    typeof ItineraryGeneratorPromptInput,
    typeof Destination,
    z.ZodTypeAny
  >('itineraryGen');
  const response = await itineraryGenerator({
    request,
    place: place.name,
    placeDescription: place.knownFor,
    activities,
  });

  const destination = response.output;
  if (!destination) {
    return null;
  }
  destination.itineraryImageUrl = place.imageUrl;
  destination.placeRef = place.ref;
  return destination;
};

export const itineraryFlow = ai.defineFlow(
  {
    name: 'itineraryFlow',
    inputSchema: ItineraryFlowInput,
    outputSchema: ItineraryFlowOutput,
  },

  async (tripDetails) => {
    const imgDescription = 'Imagine yourself high above a city where timeless elegance meets vibrant urban life. Before you stretches a breathtaking panorama, dominated by an iconic, intricate iron lattice tower that pierces the sky, a magnificent beacon inviting you to ascend for unparalleled views. Below, a majestic river gracefully carves its path through the heart of the metropolis, its shimmering waters dotted with charming boats offering scenic cruises, all while elegant stone bridges teeming with pedestrians connect the historic banks. The cityscape itself is a tapestry of classic architecture, with beautiful buildings sporting ornate facades and distinctive rooftops, interspersed with lush green trees that line the boulevards and riverbanks, hinting at serene parks and delightful walks. This destination promises an unforgettable blend of historical grandeur, romantic river strolls, and endless opportunities to immerse yourself in culture, art, and the enchanting atmosphere of a truly remarkable global city.';
    // TODO: 2. Replace the line above with this:
    // const imgDescription = await ai.run('imgDescription', async () => {
    //   if (!tripDetails.imageUrls?.length) {
    //     return '';
    //   }
    //   console.log('Generating image description...');
    //   const images = tripDetails.imageUrls.map((url) => ({
    //     media: { url },
    //   }));
    //   const response = await ai.generate({
    //     model: 'vertexai/gemini-2.5-flash',
    //     prompt: [
    //       {
    //         text: `Describe these image(s) in a detailed paragraph as though it was a tourist destination.
    // Do not give the name of the location, only give a description of what you see in the image and what you think a tourist would like it described as in a dream vacation.`,
    //       },
    //       ...images,
    //     ],
    //   });
    //   console.log('Image description generated:', response.text);
    //   return response.text;
    // });

    const places = await ai.run(
      'Retrieve matching places',
      { imgDescription, request: tripDetails.request },
      async () => {
        const docs = await ai.retrieve({
          retriever: placesRetriever,
          query: `${tripDetails.request}\n${imgDescription}`,
          options: {
            limit: 3,
          },
        });
        return docs.map((doc) => {
          const data = doc.toJSON();
          const place: Place = {
            continent: '',
            country: '',
            imageUrl: '',
            knownFor: '',
            name: '',
            ref: '',
            tags: [],
            ...data.metadata,
          };
          if (data.content[0].text) {
            place.knownFor = data.content[0].text;
          }
          delete place.embedding;
          return place;
        });
      },
    );

    const itineraries = await Promise.all(
      places.map((place, i) =>
        ai.run(`Generate itinerary #${i + 1}`, () =>
          generateItinerary(tripDetails.request, place),
        ),
      ),
    );
    return itineraries.filter((itinerary) => itinerary !== null);
  },
);
