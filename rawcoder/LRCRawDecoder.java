/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.hadoop.io.erasurecode.rawcoder;

import java.nio.ByteBuffer;
import java.util.Arrays;

import org.apache.hadoop.HadoopIllegalArgumentException;
import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.io.erasurecode.ErasureCoderOptions;
import org.apache.hadoop.io.erasurecode.rawcoder.util.DumpUtil;
import org.apache.hadoop.io.erasurecode.rawcoder.util.GF256;
import org.apache.hadoop.io.erasurecode.rawcoder.util.LRCUtil;

/**
 * A raw erasure decoder in RS code scheme in pure Java in case native one
 * isn't available in some environment. Please always use native implementations
 * when possible. This new Java coder is about 5X faster than the one originated
 * from HDFS-RAID, and also compatible with the native/ISA-L coder.
 */

 /**
  * We implemeent LRC code only with configuration 6-2-2:
  * Data units: {X0, X1, X2} {Y0, Y1, Y2}
  * Parity Units: {Px, Py, P0, P1}
  */
@InterfaceAudience.Private
public class LRCRawDecoder extends RawErasureDecoder {
  //relevant to schema and won't change during decode calls
  private byte[] encodeMatrix;

  /**
   * Below are relevant to schema and erased indexes, thus may change during
   * decode calls.
   */
  private byte[] decodeMatrix;
  private byte[] invertMatrix;
  /**
   * Array of input tables generated from coding coefficients previously.
   * Must be of size 32*k*rows
   */
  private byte[] gfTables;
  private int[] cachedErasedIndexes;
  private int[] validIndexes;
  private int numErasedDataUnits;
  private boolean[] erasureFlags;

  private int numRealInputUnits;
  private boolean localXFlag = false;
  private boolean localYFlag = false;
  private int[] localXDataIndexes;
  private int[] localYDataIndexes;

  public LRCRawDecoder(ErasureCoderOptions coderOptions) {
    super(coderOptions);

    int numAllUnits = getNumAllUnits();
    if (getNumAllUnits() >= LRCUtil.GF.getFieldSize()) {
      throw new HadoopIllegalArgumentException(
              "Invalid getNumDataUnits() and numParityUnits");
    }

    encodeMatrix = new byte[numAllUnits * getNumDataUnits()];
    LRCUtil.genEncodeMatrix(encodeMatrix, numAllUnits, getNumDataUnits());
    if (allowVerboseDump()) {
      DumpUtil.dumpMatrix(encodeMatrix, getNumDataUnits(), numAllUnits);
    }

    localXDataIndexes = new int[getNumDataUnits()/2];
    localYDataIndexes = new int[getNumDataUnits()/2];

    for (int i = 0; i < getNumDataUnits()/2; i++){
      localXDataIndexes[i] = i;
      localYDataIndexes[i] = i + getNumDataUnits()/2;
    }
  }

  @Override
  protected void doDecode(ByteBufferDecodingState decodingState) {
    CoderUtil.resetOutputBuffers(decodingState.outputs,
        decodingState.decodeLength);
    prepareDecoding(decodingState.inputs, decodingState.erasedIndexes);

    ByteBuffer[] realInputs = new ByteBuffer[numRealInputUnits];
    /**
     *  To be modified. realInputs will depend on erasedIndexes in LRC. 
     */
    for (int i = 0; i < numRealInputUnits; i++) {
      realInputs[i] = decodingState.inputs[validIndexes[i]];
    }
    LRCUtil.encodeData(gfTables, realInputs, decodingState.outputs);
  }

  @Override
  protected void doDecode(ByteArrayDecodingState decodingState) {
    int dataLen = decodingState.decodeLength;
    CoderUtil.resetOutputBuffers(decodingState.outputs,
        decodingState.outputOffsets, dataLen);
    prepareDecoding(decodingState.inputs, decodingState.erasedIndexes);

    byte[][] realInputs = new byte[numRealInputUnits][];
    int[] realInputOffsets = new int[numRealInputUnits];
    /**
     *  To be modified. realInputs will depend on erasedIndexes in LRC. 
     */
    for (int i = 0; i < this.numRealInputUnits; i++) {
      realInputs[i] = decodingState.inputs[validIndexes[i]];
      realInputOffsets[i] = decodingState.inputOffsets[validIndexes[i]];
    }
    LRCUtil.encodeData(gfTables, dataLen, realInputs, realInputOffsets,
        decodingState.outputs, decodingState.outputOffsets);
  }

  private <T> void prepareDecoding(T[] inputs, int[] erasedIndexes) {
    int[] tmpValidIndexes = CoderUtil.getValidIndexes(inputs);

    // Initialize the number of input units for global recover use
    this.numRealInputUnits = getNumDataUnits();
    int k = getNumDataUnits();
    int[] tmpRealValidIndexes = new int[getNumDataUnits()];
    // Verify if we need to recover locally or globally
    if (erasedIndexes.length == 1){
      if (erasedIndexes[0] < getNumDataUnits() + 2){
        // We only need half of data units to recover data
        this.numRealInputUnits = getNumDataUnits() / 2;

        // Create a candidate 
        int[] localIndexes = new int[this.numRealInputUnits + 1];

        if (erasedIndexes[0] < getNumDataUnits() / 2 || erasedIndexes[0] == getNumDataUnits()){
          this.localXFlag = true;

          // Generate a candidate list for local X indexes
          for (int i = 0; i < this.numRealInputUnits; i++){
            localIndexes[i] = i;
          }
          localIndexes[this.numRealInputUnits] = k;
        } // end if the first erased index is in local X part.
        else {
          this.localYFlag = true;

          // Generate a candidate list for local Y indexes
          for (int i = 0; i < this.numRealInputUnits; i++){
            localIndexes[i] = i + k / 2;
          }
          localIndexes[this.numRealInputUnits] = k + 1;
        } // end if the first erased index is in local Y part.

        // Select the local indexes from the candidate list
        tmpRealValidIndexes = new int[this.numRealInputUnits];
        int cur = 0;
        for (int i = 0; i < localIndexes.length; i++) {
          if (localIndexes[i] != erasedIndexes[0]) {
            tmpRealValidIndexes[cur++] = localIndexes[i];
          }
        }

      } // end if erasedIndexes[0] < getNumDataUnits() + 2
      else {
        this.numRealInputUnits = getNumDataUnits();
        tmpRealValidIndexes = tmpValidIndexes;
      }
    } // end if erasedIndexes.length == 1
    else if (erasedIndexes.length < getNumParityUnits()){
      this.numRealInputUnits = getNumDataUnits();
      int erasedFlag;
      if (erasedIndexes[0] < k/2 || erasedIndexes[0] == k){
        // X region has at least one erased unit
        erasedFlag = 0;
      }
      else if (erasedIndexes[0] < k || erasedIndexes[0] == k + 1){
        // Y region has at least one erased unit
        erasedFlag = 1;
      }
      else erasedFlag = 2; // All erased units are in the global parity region

      tmpRealValidIndexes = getGlobalValideIndexes(tmpValidIndexes, this.numRealInputUnits, erasedFlag);
    } // end if erasedIndexes.length < getNumParityUnits()
    else {
      if (erasedIndexesInLocal(erasedIndexes)){
        throw new HadoopIllegalArgumentException(
          "Too many erased in a local part, data not recoverable");
      }
      else {
        this.numRealInputUnits = getNumDataUnits();
        tmpRealValidIndexes = tmpValidIndexes;
      }
    }

    

    

    if (Arrays.equals(this.cachedErasedIndexes, erasedIndexes) &&
        Arrays.equals(this.validIndexes, tmpRealValidIndexes)) {
      return; // Optimization. Nothing to do
    }
    this.cachedErasedIndexes =
            Arrays.copyOf(erasedIndexes, erasedIndexes.length);
    this.validIndexes =
            Arrays.copyOf(tmpRealValidIndexes, tmpRealValidIndexes.length);

    processErasures(erasedIndexes);
  }

  private int[] getGlobalValideIndexes(int[] validIndexes, int numGlobalIndexes, int erasedFlag){
    int k = getNumDataUnits();
    int[] localXIndexes = new int[k];
    int[] localYIndexes = new int[k];
    int[] globalIndexes = new int[k];
    int curX = 0, curY = 0, curG = 0;
    for (int i = 0; i < validIndexes.length; i++){
      if (validIndexes[i] < k/2 || validIndexes[i] == k){
        localXIndexes[curX++] = validIndexes[i];
      }
      else if (validIndexes[i] < k || validIndexes[i] == k + 1){
        localYIndexes[curY++] = validIndexes[i];
      }
      else {
        globalIndexes[curG++] = validIndexes[i];
      }
    }
    int[] globalValideIndexes = new int[numGlobalIndexes];
    int cur = 0;
    if (erasedFlag == 0){
      for (int i = 0; i < curX; i++){
        globalValideIndexes[cur++] = localXIndexes[i];
      }
      for (int i = 0; i < curG; i++){
        globalValideIndexes[cur++] = globalIndexes[i];
      }
      for (int i = 0; i < curY; i++){
        globalValideIndexes[cur++] = localYIndexes[i];
      }
    }
    else if (erasedFlag == 1){
      for (int i = 0; i < curY; i++){
        globalValideIndexes[cur++] = localYIndexes[i];
      }
      for (int i = 0; i < curG; i++){
        globalValideIndexes[cur++] = globalIndexes[i];
      }
      
      for (int i = 0; i < curX; i++){
        globalValideIndexes[cur++] = localXIndexes[i];
      }
    } 
    else {
      return Arrays.copyOf(validIndexes, numGlobalIndexes);
    } 
    return Arrays.copyOf(globalValideIndexes, numGlobalIndexes);
  }

  private boolean erasedIndexesInLocal(int[] erasedIndexes){
    int[] XIndexes = new int[getNumDataUnits() / 2 + getNumParityUnits() - 1];
    int[] YIndexes = new int[getNumDataUnits() / 2 + getNumParityUnits() - 1];
    int curX = 0, curY = 0;
    for (int i = 0; i < getNumDataUnits() / 2; i++){
      XIndexes[curX++] = i;
      YIndexes[curY++] = i + getNumDataUnits() / 2;
    }
    XIndexes[curX++] = getNumDataUnits();
    YIndexes[curY++] = getNumDataUnits() + 1;
    for (int i = getNumAllUnits() - 2; i < getNumAllUnits(); i++){
      XIndexes[curX++] = i;
      YIndexes[curY++] = i;
    }
    
    boolean inLocalX = contain(erasedIndexes, XIndexes);
    boolean inLocalY = contain(erasedIndexes, YIndexes);
    if(inLocalX == true || inLocalY == true){
      return true;
    }
    else{
      return false;
    }

  }

  private boolean contain(int[] X, int[] Y){
    /**
     * Return a boolean indicating if X is contained in Y
     * i.e. all elements of X belong to Y
     */
    boolean allInY = true;
    for (int i = 0; i < X.length; i++){
      boolean elementInY = false;
      for (int j = 0; j < Y.length; j++){
        if (X[i] == Y[j]){
          elementInY = true;
          break;
        }
      }
      if (!elementInY) {
        allInY = false;
        break;
      }
    }
    return allInY;
  }

  private void processErasures(int[] erasedIndexes) {

    this.decodeMatrix = new byte[getNumAllUnits() * this.numRealInputUnits];
    this.invertMatrix = new byte[getNumAllUnits() * this.numRealInputUnits];
    this.gfTables = new byte[getNumAllUnits() * this.numRealInputUnits * 32];

    this.erasureFlags = new boolean[getNumAllUnits()];
    this.numErasedDataUnits = 0;

    for (int i = 0; i < erasedIndexes.length; i++) {
      int index = erasedIndexes[i];
      erasureFlags[index] = true;
      if (index < getNumDataUnits()) {
        numErasedDataUnits++;
      }
    }

    generateDecodeMatrix(erasedIndexes);

    LRCUtil.initTables(getNumDataUnits(), erasedIndexes.length,
        decodeMatrix, 0, gfTables);
    if (allowVerboseDump()) {
      System.out.println(DumpUtil.bytesToHex(gfTables, -1));
    }
  }

  // Generate decode matrix from encode matrix
  private void generateDecodeMatrix(int[] erasedIndexes) {
    int i, j, r, p;
    int[] realInputIndexes;
    byte s;
    byte[] tmpMatrix = new byte[getNumAllUnits() * this.numRealInputUnits];

    

    // Construct matrix tmpMatrix by removing error rows
    for (i = 0; i < this.numRealInputUnits; i++) {
      /** To be modified, here it picks the first getNumDataUnits() 
       * valide units for decoding. However, in LRC the valide units
       * for decoding depend on the erasedIndexes. 
       */
      r = validIndexes[i]; 

      for (j = 0; j < this.numRealInputUnits; j++) {
        tmpMatrix[this.numRealInputUnits * i + j] =
                encodeMatrix[getNumDataUnits() * r + j];
      }
    }

    GF256.gfInvertMatrix(tmpMatrix, invertMatrix, this.numRealInputUnits);

    for (i = 0; i < numErasedDataUnits; i++) {
      for (j = 0; j < this.numRealInputUnits; j++) {
        decodeMatrix[this.numRealInputUnits * i + j] =
                invertMatrix[getNumDataUnits() * erasedIndexes[i] + j];
      }
    }

    if (localXFlag || localYFlag){
      realInputIndexes = localXFlag ? this.localXDataIndexes : this.localYDataIndexes;

      for (p = numErasedDataUnits; p < erasedIndexes.length; p++) {
        for (i = 0; i < this.numRealInputUnits; i++) {
          s = 0;
          for (j = 0; j < this.numRealInputUnits; j++) {
            s ^= GF256.gfMul(invertMatrix[realInputIndexes[j] * this.numRealInputUnits + i],
                    encodeMatrix[getNumDataUnits() * erasedIndexes[p] + realInputIndexes[j]]);
          }
          decodeMatrix[this.numRealInputUnits * p + i] = s;
        }
      }
    }

    for (p = numErasedDataUnits; p < erasedIndexes.length; p++) {
      for (i = 0; i < this.numRealInputUnits; i++) {
        s = 0;
        for (j = 0; j < this.numRealInputUnits; j++) {
          s ^= GF256.gfMul(invertMatrix[j * this.numRealInputUnits + i],
                  encodeMatrix[getNumDataUnits() * erasedIndexes[p] + j]);
        }
        decodeMatrix[this.numRealInputUnits * p + i] = s;
      }
    }
  }
}
